"""Train residual DeepFM on rolling OOF LightGBM Top-200, select on W5.

First run: python rank/rolling_oof_v6.py
Then run:  python rank/deepfm_residual_v2.py

W1-W4 are out-of-fold fine-train snapshots. W5 is an independent
chronological fine-selection snapshot. Original valid is diagnostic only,
as the final upstream LightGBM was tuned on valid. Run --final-test only
once the full pipeline is frozen. Do not tune after inspecting test.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from rolling_oof_v6 import read, write, merge_recall, ROOT, CHANNELS

CHFEATS = [f'{ch}_{suffix}' for ch in CHANNELS for suffix in ('hit', 'rank', 'norm_score')]
SPARSE = ['user_id', 'note_id', 'topic_id']
DENSE = (['user_hist_log', 'note_pop_log', 'topic_affinity', 'age_days',
          'lgbm_logit', 'coarse_rank_pct', 'rrf_score', 'channel_count'] + CHFEATS)


def topic_ids(notes):
    if 'tags' in notes.columns:
        tag = notes.tags.fillna('').astype(str).str.split(',').str[0]
    else:
        tag = notes.category.astype(str)
    keys = notes.category.astype(str) + '::' + tag
    ids, unique = pd.factorize(keys, sort=True)
    return ids.astype('int64'), list(unique)


def add_features(candidates, history, notes, cutoff, topics):
    """Recreate snapshot-dependent features. Never use future labels here."""
    meta = notes[['note_id', 'publish_ts']].copy()
    meta['topic_id'] = topics
    frame = candidates.drop(columns=['publish_ts', 'topic_id'], errors='ignore').merge(
        meta, on='note_id', how='left', validate='many_to_one')
    if frame.publish_ts.isna().any():
        raise ValueError('Unknown note_id in candidates')
    frame = frame.loc[frame.publish_ts <= cutoff].copy()
    seen = history[['user_id', 'note_id']].drop_duplicates()
    frame = frame.loc[~pd.MultiIndex.from_frame(frame[['user_id','note_id']]).isin(
        pd.MultiIndex.from_frame(seen))].copy()
    h = seen.merge(meta[['note_id', 'topic_id']], on='note_id', how='left')
    uc = h.groupby('user_id').size()
    ic = h.groupby('note_id').size()
    tc = h.groupby(['user_id','topic_id']).size()
    user_counts = frame.user_id.map(uc).fillna(0).to_numpy('float32')
    frame['user_hist_log'] = np.log1p(user_counts)
    frame['note_pop_log'] = np.log1p(frame.note_id.map(ic).fillna(0).to_numpy('float32'))
    keys = pd.MultiIndex.from_frame(frame[['user_id','topic_id']])
    frame['topic_affinity'] = (tc.reindex(keys, fill_value=0).to_numpy('float32')
                               / np.maximum(user_counts, 1))
    frame['age_days'] = ((cutoff - frame.publish_ts)/86400).astype('float32')
    p = np.clip(frame.prediction.to_numpy('float64'), 1e-6, 1-1e-6)
    frame['lgbm_logit'] = np.log(p/(1-p)).astype('float32')
    frame['coarse_rank_pct'] = (frame.coarse_rank.to_numpy('float32') / 200.0)
    for ch in CHANNELS:
        r = f'{ch}_rank'
        # Canonical missing rank of 10000 is mapped to 1.1 instead of dominating std.
        frame[r] = np.where(frame[f'{ch}_hit'].to_numpy() > 0,
                            np.log1p(frame[r].to_numpy('float32')) / np.log1p(200),
                            1.1).astype('float32')
    missing = [c for c in DENSE + SPARSE if c not in frame]
    if missing:
        raise ValueError(f'Fine features missing: {missing}')
    return frame


def truth_at_snapshot(target, history, notes, cutoff):
    positives = target[['user_id','note_id']].drop_duplicates()
    eligible = notes.publish_ts.to_numpy() <= cutoff
    positives = positives.loc[eligible[positives.note_id.to_numpy('int64')]]
    seen = pd.MultiIndex.from_frame(history[['user_id','note_id']].drop_duplicates())
    positives = positives.loc[~pd.MultiIndex.from_frame(positives).isin(seen)]
    return positives.groupby('user_id').note_id.agg(set).to_dict()


def get_pairs(frame, rng, pairs_per_positive):
    positives = frame.loc[frame.label == 1].copy().reset_index(drop=True)
    negative = frame.loc[frame.label == 0].copy().reset_index(drop=True)
    neg_groups = negative.groupby('user_id').indices
    p_idx, n_idx = [], []
    for idx, uid in enumerate(positives.user_id.to_numpy('int64')):
        group = neg_groups.get(int(uid))
        if group is None or not len(group):
            continue
        neg_samples = rng.choice(group, size=pairs_per_positive, replace=True)
        p_idx.extend([idx] * pairs_per_positive)
        n_idx.extend(neg_samples.tolist())
    if not p_idx:
        raise ValueError('No within-user positive-negative pairs')
    return positives.iloc[p_idx].copy(), negative.iloc[n_idx].copy()


class ResidualDeepFMv2(nn.Module):
    def __init__(self, cardinalities, dense_size, dim=12, max_delta=1.5):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, dim) for c in cardinalities])
        self.first_order = nn.ModuleList([nn.Embedding(c, 1) for c in cardinalities])
        self.dense_first = nn.Linear(dense_size, 1)
        self.trunk = nn.Sequential(
            nn.Linear(len(cardinalities)*dim + dense_size, 128), nn.ReLU(),
            nn.Dropout(.20), nn.Linear(128, 64), nn.ReLU())
        self.residual_head = nn.Linear(64 + 2, 1)
        self.max_delta = max_delta
        for emb in self.emb:
            nn.init.normal_(emb.weight, std=.015)
        for emb in self.first_order:
            nn.init.zeros_(emb.weight)
        nn.init.zeros_(self.dense_first.weight)
        nn.init.zeros_(self.dense_first.bias)
        # Exact zero residual initially; the head receives gradients immediately.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, sparse, dense, coarse_logit, multiplier=1.0, return_delta=False):
        x = torch.stack([e(sparse[:,i]) for i,e in enumerate(self.emb)], dim=1)
        first = sum(e(sparse[:,i]) for i,e in enumerate(self.first_order)) + self.dense_first(dense)
        s = x.sum(1)
        fm = .5 * (s.square() - x.square().sum(1)).sum(1, keepdim=True)
        deep = self.trunk(torch.cat([x.flatten(1),dense],dim=1))
        pre = self.residual_head(torch.cat([deep,first,fm],dim=1)).squeeze(1)
        delta = self.max_delta * multiplier * torch.tanh(pre)
        return (coarse_logit, delta) if return_delta else (coarse_logit + delta)


def tensors(frame, mean, std):
    sparse = torch.as_tensor(frame[SPARSE].to_numpy('int64',copy=True))
    dense = frame[DENSE].to_numpy('float32',copy=True)
    dense = np.nan_to_num((dense - mean) / std, nan=0., posinf=0., neginf=0.)
    dense = np.clip(dense, -8, 8)
    lg = frame.lgbm_logit.to_numpy('float32',copy=True)
    return sparse, torch.from_numpy(dense), torch.from_numpy(lg)


@torch.no_grad()
def infer(model, frame, mean, std, device, batch_size, multiplier):
    model.eval()
    sp,de,base=tensors(frame,mean,std)
    result=[]
    for st in range(0,len(sp),batch_size):
        logits = model(sp[st:st+batch_size].to(device),
                       de[st:st+batch_size].to(device),
                       base[st:st+batch_size].to(device), multiplier=multiplier)
        result.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(result) if result else np.empty(0,'float32')


def evaluate(frame, truth, score_col='fine_score', k=20):
    ranked = (frame.sort_values(['user_id', score_col,'note_id'], ascending=[True,False,True])
              .groupby('user_id',sort=False).head(k))
    group = ranked.groupby('user_id').note_id.agg(list).to_dict()
    recalls=[]; ndcgs=[]
    for uid, gt in truth.items():
        rec=group.get(uid,[])
        hits=np.fromiter((int(i in gt) for i in rec),dtype='float64')
        recalls.append(hits.sum()/len(gt))
        dcg=(hits/np.log2(np.arange(len(hits))+2)).sum()
        idcg=(1/np.log2(np.arange(min(k,len(gt)))+2)).sum()
        ndcgs.append(dcg/idcg)
    return (float(np.mean(recalls)) if recalls else 0.0,
            float(np.mean(ndcgs)) if ndcgs else 0.0)


def official_frame(src, coarse_dir, notes, topics, split, include_truth=False):
    train=read(src,'train')
    valid=read(src,'valid')
    history = train if split=='valid' else pd.concat([train, valid],ignore_index=True)
    cutoff=int(history.timestamp.max())
    raw = read(coarse_dir, f'coarse_{split}_top200')
    merged = merge_recall(src, split=split)
    columns = ['user_id','note_id','rrf_score','channel_count'] + CHFEATS
    # Note: merge_recall() also contains the five original scores, but the v2
    # fine tower deliberately uses normalized per-channel features and ranks.
    raw = raw.merge(merged[columns], on=['user_id','note_id'], how='left', validate='one_to_one')
    if raw[CHFEATS].isna().any().any():
        raise ValueError('Coarse Top-200 contains candidate absent from five recall tables')
    frame=add_features(raw,history,notes,cutoff,topics)
    truth=(truth_at_snapshot(read(src,split),history,notes,cutoff)
           if include_truth else {})
    return frame,truth


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=ROOT/'data'/'raw_v6')
    ap.add_argument('--coarse-dir',type=Path,default=ROOT/'rank'/'output_v6')
    ap.add_argument('--oof-dir',type=Path,default=ROOT/'rank'/'output_v6'/'rolling_oof')
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--batch-size',type=int,default=4096)
    ap.add_argument('--lr',type=float,default=.0002)
    ap.add_argument('--emb-dim',type=int,default=12)
    ap.add_argument('--max-delta',type=float,default=1.5)
    ap.add_argument('--pair-negatives',type=int,default=8)
    ap.add_argument('--residual-regularizer',type=float,default=.02)
    ap.add_argument('--patience',type=int,default=3)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--final-test',action='store_true',
                    help='Evaluate TEST once, after all design and hyperparameters are frozen')
    args=ap.parse_args()
    rng=np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    notes=read(args.data,'notes')
    users=read(args.data,'users')
    if not np.array_equal(notes.note_id.to_numpy(),np.arange(len(notes))):
        raise ValueError('note_id must be contiguous from 0')
    if not np.array_equal(users.user_id.to_numpy(),np.arange(len(users))):
        raise ValueError('user_id must be contiguous from 0')
    topics,topic_keys=topic_ids(notes)
    pos_parts,neg_parts=[] , []
    for i in range(1,5):
        snap=args.oof_dir/f'fold_{i:02d}'
        hist=read(snap,'rank_train_history')
        cutoff=int(hist.timestamp.max())
        df=read(args.oof_dir,f'oof_fold_{i:02d}_top200')
        p,n=get_pairs(df,rng,args.pair_negatives)
        pos_parts.append(add_features(p,hist,notes,cutoff,topics))
        neg_parts.append(add_features(n,hist,notes,cutoff,topics))
        print(f'OOF W{i}: positive pairs {len(p):,}; negative pairs {len(n):,}',flush=True)
    pos=pd.concat(pos_parts,ignore_index=True)
    neg=pd.concat(neg_parts,ignore_index=True)
    if len(pos)!=len(neg):
        raise AssertionError('Pair counts differ')
    h5=read(args.oof_dir/'fold_05','rank_train_history')
    cutoff5=int(h5.timestamp.max())
    w5=read(args.oof_dir,'oof_fold_05_top200')
    fold5=add_features(w5,h5,notes,cutoff5,topics)
    truth5=truth_at_snapshot(read(args.oof_dir,'target_fold_05'),h5,notes,cutoff5)
    if not truth5:
        raise ValueError('Empty W5 truth')
    mean=pd.concat([pos[DENSE],neg[DENSE]],ignore_index=True).to_numpy('float32').mean(0)
    std=pd.concat([pos[DENSE],neg[DENSE]],ignore_index=True).to_numpy('float32').std(0)
    std=np.maximum(std,1e-5)
    sp,dp,bp=tensors(pos,mean,std)
    sn,dn,bn=tensors(neg,mean,std)
    loader=DataLoader(TensorDataset(sp,dp,bp,sn,dn,bn),batch_size=args.batch_size,
                      shuffle=True,pin_memory=(device.type=='cuda'))
    model=ResidualDeepFMv2([len(users),len(notes),len(topic_keys)],len(DENSE),
                            args.emb_dim,args.max_delta).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=5e-5)
    candidate_scales=[0.0,.25,.50,1.0]
    baseline=evaluate(fold5,truth5,'prediction',20)
    print(f'W5 coarse baseline Recall@20={baseline[0]:.6f} NDCG@20={baseline[1]:.6f}',flush=True)
    best_score=baseline[1]
    best_weight=0.0
    best_epoch=0
    best_state=copy.deepcopy({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    failed=0
    for epoch in range(1,args.epochs+1):
        model.train()
        total=0.0
        for pu,p_dense,pl,nu,n_dense,nl in loader:
            pu,p_dense,pl=pu.to(device),p_dense.to(device),pl.to(device)
            nu,n_dense,nl=nu.to(device),n_dense.to(device),nl.to(device)
            pc,delta_p=model(pu,p_dense,pl,return_delta=True)
            nc,delta_n=model(nu,n_dense,nl,return_delta=True)
            rank_loss=F.softplus(-((pc+delta_p)-(nc+delta_n))).mean()
            reg=(delta_p.square().mean()+delta_n.square().mean())/2
            loss=rank_loss+args.residual_regularizer*reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total+=float(loss.detach())*len(pu)
        for alpha in candidate_scales:
            if alpha==0.:
                ndcg=baseline[1];recall=baseline[0]
            else:
                fold5['fine_score']=infer(model,fold5,mean,std,device,args.batch_size,alpha)
                recall,ndcg=evaluate(fold5,truth5,'fine_score',20)
            print(f'Epoch={epoch} scale={alpha:.2f} Recall@20={recall:.6f} NDCG@20={ndcg:.6f}',flush=True)
            if ndcg>best_score+1e-8:
                best_score=ndcg
                best_epoch=epoch
                best_weight=alpha
                best_state=copy.deepcopy({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
                failed=0
        if best_epoch!=epoch:
            failed+=1
            if failed>=args.patience:
                print('Early stopping',flush=True)
                break
    model.load_state_dict(best_state)
    checkpoint=args.coarse_dir/'deepfm_residual_v2.pt'
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':best_state,'dense':DENSE,'sparse':SPARSE,
                'mean':mean,'std':std,'topic_keys':topic_keys,
                'cardinalities':[len(users),len(notes),len(topic_keys)],
                'max_delta':args.max_delta,'scale':best_weight,
                'best_epoch':best_epoch,'W5_NDCG20':best_score}, checkpoint)
    print(f'Selected epoch={best_epoch} multiplier={best_weight}; W5 NDCG@20={best_score:.6f}',flush=True)
    # Do not use original valid/test labels for model fitting or selection.
    for split in ('valid','test'):
        frame,truth=official_frame(args.data,args.coarse_dir,notes,topics,split,
                                   include_truth=(split=='valid' or args.final_test))
        fine=infer(model,frame,mean,std,device,args.batch_size,best_weight)
        output=frame[['user_id','note_id','prediction','coarse_rank']].copy()
        output['fine_score']=fine
        output=output.sort_values(['user_id','fine_score','note_id'],ascending=[True,False,True]).reset_index(drop=True)
        output['fine_rank']=output.groupby('user_id').cumcount().add(1).astype('int32')
        dest=write(output,args.coarse_dir,f'fine_{split}_top200_v2')
        print(f'Exported {split}: {len(output):,} rows to {dest}',flush=True)
        if split=='valid' or args.final_test:
            for k in (20,100,200):
                base_r,base_n=evaluate(output,truth,'prediction',k)
                new_r,new_n=evaluate(output,truth,'fine_score',k)
                print(f'{split} K={k}: coarse R={base_r:.6f} NDCG={base_n:.6f} | '
                      f'DeepFM R={new_r:.6f} NDCG={new_n:.6f}',flush=True)
    print('Done. Test metrics are only printed with --final-test. Do not retune on test.')

if __name__=='__main__':
    main()
