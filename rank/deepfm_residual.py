"""Temporal cross-fit LightGBM -> residual DeepFM for v6.

Run: python rank/deepfm_residual_v6.py
Requires existing five *_train files and coarse_valid/test_top200 files.
A/B/C are chronological thirds of rank_train_future. Coarse fitted only on A;
DeepFM fitted only on B and selected only on C. Thus fine train/selection
never see an in-sample coarse prediction. Test uses the existing final coarse model.
Caveat: output for 'valid' is diagnostic only: upstream LightGBM was selected
on valid labels. Use test once after fixing hyperparameters.
"""
import argparse
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
# Self-contained: feature merging and label engineering match the v6 coarse baseline.
CHANNELS = {'MostPopular':'mostpopular_decay', 'ItemCF':'itemcf_weighted',
            'BPR-MF':'bprmf', 'LightGCN':'lightgcn', 'TwoTower':'twotower'}
TOPKS = (100, 200)

def read(root, stem):
    for ext in ('.parquet', '.csv'):
        p = root / (stem + ext)
        if p.exists():
            return pd.read_parquet(p) if ext == '.parquet' else pd.read_csv(p)
    raise FileNotFoundError(f'缺少文件: {root}/{stem}.parquet 或 .csv')

def check_inputs(root):
    for stem in ('notes', 'train', 'valid', 'test', 'rank_train_history', 'rank_train_future'):
        if not any((root/f'{stem}{ext}').exists() for ext in ('.parquet','.csv')):
            raise FileNotFoundError(f'缺少 {stem}.parquet/.csv')
    missing=[]
    for split in ('train','valid','test'):
        for p in CHANNELS.values():
            if not any((root/f'{p}_{split}{ext}').exists() for ext in ('.parquet','.csv')):
                missing.append(f'{p}_{split}')
    if missing:
        raise FileNotFoundError('请先导出这些五路召回候选（train 必须由历史前缀模型生成）:\n'+'\n'.join(missing))

def merge_channels(root, split):
    base=None
    for name, prefix in CHANNELS.items():
        raw=read(root, f'{prefix}_{split}')
        if 'recall_score' in raw.columns and 'score' not in raw.columns:
            raw=raw.rename(columns={'recall_score':'score'})
        if not {'user_id','note_id','score'}.issubset(raw):
            raise ValueError(f'{prefix}_{split} 必须有 user_id,note_id,score')
        x=raw[['user_id','note_id','score']].copy()
        x=x.dropna(subset=['score'])
        x=x.loc[np.isfinite(x['score'])]
        x=x.sort_values(['user_id','score','note_id'], ascending=[True,False,True])
        x=x.drop_duplicates(['user_id','note_id'])
        x['rank']=x.groupby('user_id').cumcount().add(1).astype('int32')
        # normalize within user/channel; never combine raw scores from different models
        agg=x.groupby('user_id')['score']
        lo=agg.transform('min'); hi=agg.transform('max')
        x['norm_score']=((x['score']-lo)/(hi-lo).replace(0,np.nan)).fillna(1.0)
        x[f'{name}_hit']=np.int8(1)
        x[f'{name}_rrf']=1.0/(60.0+x['rank'])
        x=x.rename(columns={'score':f'{name}_score','rank':f'{name}_rank',
                            'norm_score':f'{name}_norm_score'})
        x=x.set_index(['user_id','note_id'])
        base=x if base is None else base.join(x,how='outer')
    df=base.reset_index()
    for name in CHANNELS:
        for suffix in ('hit','rrf','norm_score'):
            col=f'{name}_{suffix}'
            df[col]=df[col].fillna(0)
        df[f'{name}_rank']=df[f'{name}_rank'].fillna(10000)
    df['channel_count']=df[[f'{x}_hit' for x in CHANNELS]].sum(axis=1)
    df['rrf_score']=df[[f'{x}_rrf' for x in CHANNELS]].sum(axis=1)
    df['max_norm_score']=df[[f'{x}_norm_score' for x in CHANNELS]].max(axis=1)
    df['mean_norm_score']=df[[f'{x}_norm_score' for x in CHANNELS]].sum(axis=1)/df.channel_count.clip(lower=1)
    return df

def make_samples(candidates, history, future, notes, cutoff):
    if not history.empty and history.timestamp.max()>cutoff:
        raise ValueError('历史含有推荐时刻之后的数据')
    meta=notes[['note_id','category','publish_ts']].copy()
    df=candidates.merge(meta,on='note_id',how='left',validate='many_to_one')
    if df.publish_ts.isna().any(): raise ValueError('候选含不存在的 note_id')
    df=df.loc[df.publish_ts<=cutoff].copy()
    h=history[['user_id','note_id']].drop_duplicates()
    old=pd.MultiIndex.from_frame(h)
    df=df.loc[~pd.MultiIndex.from_frame(df[['user_id','note_id']]).isin(old)].copy()
    hs=h.merge(meta[['note_id','category']],on='note_id',how='left')
    u=hs.groupby('user_id').size()
    n=hs.groupby('note_id').size()
    uc=hs.groupby(['user_id','category']).size()
    df['user_hist_log']=np.log1p(df.user_id.map(u).fillna(0).to_numpy())
    df['note_pop_log']=np.log1p(df.note_id.map(n).fillna(0).to_numpy())
    idx=pd.MultiIndex.from_frame(df[['user_id','category']])
    df['cat_affinity']=np.asarray(uc.reindex(idx,fill_value=0),dtype=np.float32)/np.maximum(np.expm1(df.user_hist_log.to_numpy()),1)
    df['note_age_days']=((cutoff-df.publish_ts)/86400).astype('float32')
    # All relevant future positives visible at cutoff, excluding history
    truth=future[['user_id','note_id']].drop_duplicates()
    truth=truth.merge(meta[['note_id','publish_ts']],on='note_id',how='left')
    truth=truth.loc[truth.publish_ts<=cutoff,['user_id','note_id']]
    truth=truth.loc[~pd.MultiIndex.from_frame(truth).isin(old)].copy()
    df['label']=pd.MultiIndex.from_frame(df[['user_id','note_id']]).isin(pd.MultiIndex.from_frame(truth)).astype('int8')
    df['category']=df.category.astype('category')
    return df,truth

def metrics(df,truth,score,k):
    ranked=df.sort_values(['user_id',score,'note_id'],ascending=[True,False,True]).groupby('user_id',sort=False).head(k)
    rec=ranked.groupby('user_id').note_id.agg(list).to_dict()
    ground=truth.groupby('user_id').note_id.agg(set).to_dict()
    recalls=[]; ndcgs=[]
    for uid, items in ground.items():
        hits=np.array([int(i in items) for i in rec.get(uid,[])],dtype=float)
        recalls.append(hits.sum()/len(items))
        dcg=(hits/np.log2(np.arange(len(hits))+2)).sum()
        ideal=(1/np.log2(np.arange(min(k,len(items)))+2)).sum()
        ndcgs.append(dcg/ideal)
    return float(np.mean(recalls)),float(np.mean(ndcgs))


SPARSE = ['user_id', 'note_id', 'topic_id']
CHANNEL_FEATURES = [f'{ch}_{part}' for ch in CHANNELS for part in ('hit', 'rank', 'norm_score')]
DENSE = ['user_hist_log', 'note_pop_log', 'topic_affinity', 'age_days',
         'lgbm_logit', 'lgbm_rank_pct', 'rrf_score', 'channel_count'] + CHANNEL_FEATURES


def read(root, stem):
    for ext in ('.parquet', '.csv'):
        p = root / f'{stem}{ext}'
        if p.exists():
            return pd.read_parquet(p) if ext == '.parquet' else pd.read_csv(p)
    raise FileNotFoundError(f'Missing {stem}.parquet/.csv under {root}')


def split_future(future):
    """Three sequential, non-overlapping event windows A/B/C."""
    x = future.sort_values('timestamp').reset_index(drop=True)
    if len(x) < 30:
        raise ValueError('Too few future interactions for A/B/C split')
    t1, t2 = (int(x.timestamp.quantile(q)) for q in (1/3, 2/3))
    a = x[x.timestamp <= t1].copy()
    b = x[(x.timestamp > t1) & (x.timestamp <= t2)].copy()
    c = x[x.timestamp > t2].copy()
    if any(part.empty for part in (a,b,c)):
        raise ValueError('Cannot split future into three non-empty time windows')
    if not (a.timestamp.max() < b.timestamp.min() and b.timestamp.max() < c.timestamp.min()):
        raise AssertionError('A/B/C chronology violated')
    return a,b,c


def topic_ids_for_notes(notes):
    tags = (notes.tags.fillna('').astype(str).str.split(',').str[0]
            if 'tags' in notes else notes.category.astype(str))
    keys = notes.category.astype(str) + '::' + tags
    codes, uniques = pd.factorize(keys, sort=True)
    return codes.astype('int64'), list(uniques)


def add_deep_features(df, hist, notes, cutoff, topic_ids):
    """Use only observed historical interactions and public note metadata."""
    meta = notes[['note_id', 'publish_ts']].copy()
    meta['topic_id'] = topic_ids
    out = df.drop(columns=['publish_ts', 'topic_id'], errors='ignore').copy()
    out = out.merge(meta, on='note_id', how='left', validate='many_to_one')
    if out.publish_ts.isna().any():
        raise ValueError('Unknown note_id')
    out = out.loc[out.publish_ts <= cutoff].copy()
    seen = hist[['user_id','note_id']].drop_duplicates()
    out = out.loc[~pd.MultiIndex.from_frame(out[['user_id','note_id']]).isin(
        pd.MultiIndex.from_frame(seen))].copy()
    h = seen.merge(meta[['note_id','topic_id']], on='note_id', how='left')
    uc = h.groupby('user_id').size()
    ic = h.groupby('note_id').size()
    tc = h.groupby(['user_id','topic_id']).size()
    ucount = out.user_id.map(uc).fillna(0).to_numpy('float32')
    out['user_hist_log'] = np.log1p(ucount)
    out['note_pop_log'] = np.log1p(out.note_id.map(ic).fillna(0).to_numpy('float32'))
    keys = pd.MultiIndex.from_frame(out[['user_id','topic_id']])
    out['topic_affinity'] = np.asarray(tc.reindex(keys, fill_value=0), dtype='float32') / np.maximum(ucount,1)
    out['age_days'] = ((cutoff-out.publish_ts)/86400).to_numpy('float32')
    return out


def add_coarse_scores(df, booster, features, ntrees, categories, topk):
    frame = df.copy()
    frame['category'] = pd.Categorical(frame['category'], categories=categories)
    p = np.asarray(booster.predict(frame[features], num_iteration=ntrees), dtype='float64')
    frame['prediction'] = p
    return filter_topk(frame, 'prediction', topk)


def filter_topk(df, col, topk):
    x = (df.sort_values(['user_id',col,'note_id'], ascending=[True,False,True])
           .groupby('user_id', sort=False).head(topk).copy())
    x['coarse_rank'] = x.groupby('user_id').cumcount().add(1)
    return x.reset_index(drop=True)


def finalize(df, history, notes, cutoff, topics, labels, coarse_col='prediction'):
    x = add_deep_features(df, history, notes, cutoff, topics)
    if labels is not None:
        # Restrict positives to notes visible at the SAME fixed snapshot.
        truth = labels[['user_id','note_id']].drop_duplicates()
        eligible = notes.set_index('note_id').publish_ts
        truth = truth[truth.note_id.map(eligible) <= cutoff]
        seen = pd.MultiIndex.from_frame(history[['user_id','note_id']].drop_duplicates())
        truth = truth.loc[~pd.MultiIndex.from_frame(truth).isin(seen)].copy()
        x['label'] = pd.MultiIndex.from_frame(x[['user_id','note_id']]).isin(
            pd.MultiIndex.from_frame(truth)).astype('float32')
        truth_map = truth.groupby('user_id').note_id.agg(set).to_dict()
    else:
        truth_map = {}
    probs = np.clip(x[coarse_col].to_numpy('float64'),1e-6,1-1e-6)
    x['lgbm_logit'] = np.log(probs/(1-probs)).astype('float32')
    x['lgbm_rank_pct'] = (x['coarse_rank'].to_numpy('float32')/200.0)
    # RRF may be absent from a previous coarse Top200 export.
    if 'rrf_score' not in x:
        x['rrf_score'] = 0.0
    if 'channel_count' not in x:
        x['channel_count'] = 0.0
    missing=[col for col in CHANNEL_FEATURES if col not in x]
    if missing:
        raise ValueError(f'Missing channel features: {missing}')
    return x.reset_index(drop=True), truth_map


class ResidualDeepFM(nn.Module):
    def __init__(self, dims, dense_dim, emb_dim=12, residual_scale=0.5):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(n,emb_dim) for n in dims])
        self.linear = nn.ModuleList([nn.Embedding(n,1) for n in dims])
        self.dense_linear = nn.Linear(dense_dim,1)
        in_dim = len(dims)*emb_dim+dense_dim
        self.mlp = nn.Sequential(nn.Linear(in_dim,96),nn.ReLU(),nn.Dropout(.35),
                                 nn.Linear(96,48),nn.ReLU(),nn.Dropout(.25),nn.Linear(48,1))
        self.residual_scale = residual_scale
        self.rho = nn.Parameter(torch.tensor(0.0))
        for e in self.emb: nn.init.normal_(e.weight,std=.015)
        for e in self.linear: nn.init.zeros_(e.weight)

    def forward(self, cat, dense, base_logit):
        emb = torch.stack([m(cat[:,j]) for j,m in enumerate(self.emb)],dim=1)
        first = sum(m(cat[:,j]) for j,m in enumerate(self.linear)) + self.dense_linear(dense)
        summed = emb.sum(1)
        fm = .5*(summed.square()-emb.square().sum(1)).sum(1,keepdim=True)
        deep = self.mlp(torch.cat([emb.flatten(1),dense],dim=1))
        delta = torch.tanh((first+fm+deep).squeeze(1))
        # Initially score == upstream LightGBM logit; bounded correction.
        return base_logit + self.residual_scale * torch.tanh(self.rho) * delta


def make_tensors(df, mean, std):
    sp = torch.from_numpy(df[SPARSE].to_numpy('int64',copy=True))
    raw = df[DENSE].to_numpy('float32',copy=True)
    de = torch.from_numpy(np.nan_to_num((raw-mean)/std,nan=0.0,posinf=0.0,neginf=0.0))
    base = torch.from_numpy(df.lgbm_logit.to_numpy('float32',copy=True))
    return sp,de,base


@torch.no_grad()
def infer(model, df, mean, std, device, bs):
    model.eval(); sp,de,base=make_tensors(df,mean,std)
    out=[]
    for k in range(0,len(df),bs):
        out.append(torch.sigmoid(model(sp[k:k+bs].to(device),de[k:k+bs].to(device),
                                         base[k:k+bs].to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0,'float32')


def topk_metric(df, truth, score='fine_score', k=20):
    x = filter_topk(df,score,k)
    rec = x.groupby('user_id').note_id.agg(list).to_dict()
    recalls=[]; ndcgs=[]
    for uid,gt in truth.items():
        items=rec.get(uid,[])
        hits=np.fromiter((i in gt for i in items),dtype='float64')
        recalls.append(hits.sum()/len(gt))
        dcg=(hits/np.log2(np.arange(len(hits))+2)).sum()
        ideal=(1/np.log2(np.arange(min(k,len(gt)))+2)).sum()
        ndcgs.append(dcg/ideal)
    return (float(np.mean(recalls)) if recalls else 0.0,
            float(np.mean(ndcgs)) if ndcgs else 0.0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=ROOT/'data'/'raw_v6')
    p.add_argument('--coarse-dir',type=Path,default=ROOT/'rank'/'output_v6')
    p.add_argument('--output',type=Path,default=ROOT/'rank'/'output_v6')
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=4096)
    p.add_argument('--negative-ratio',type=int,default=15)
    p.add_argument('--lr',type=float,default=0.0005)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--topk',type=int,default=200)
    p.add_argument('--coarse-trees',type=int,default=100)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.topk != 200: raise ValueError('Current feature normalization expects --topk=200')
    rng=np.random.default_rng(args.seed);torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    users=read(args.data,'users');notes=read(args.data,'notes')
    h=read(args.data,'rank_train_history');future=read(args.data,'rank_train_future')
    train=read(args.data,'train');valid=read(args.data,'valid');test=read(args.data,'test')
    if not np.array_equal(notes.note_id.to_numpy(),np.arange(len(notes))): raise ValueError('note_id must be contiguous')
    if not np.array_equal(users.user_id.to_numpy(),np.arange(len(users))): raise ValueError('user_id must be contiguous')
    a,b,c=split_future(future)
    print(f'Event-time A/B/C sizes: {len(a):,} / {len(b):,} / {len(c):,}',flush=True)
    # Fixed historical candidate snapshot; no B/C labels involved in candidate generation.
    merged=merge_channels(args.data,'train')
    base, _=make_samples(merged,h,a,notes,int(h.timestamp.max()))
    coarse_features=[z for z in base.columns if z not in {'user_id','note_id','publish_ts','label','prediction'}]
    cats=sorted(notes.category.astype(str).unique())
    base['category']=pd.Categorical(base.category,categories=cats)
    if base.label.sum()==0:raise ValueError('No coarse positives in A; cannot train temporary coarse model')
    pos=np.flatnonzero(base.label.to_numpy()==1); neg=np.flatnonzero(base.label.to_numpy()==0)
    nneg=min(len(neg),len(pos)*20)
    coarse_fit=base.iloc[np.r_[pos,rng.choice(neg,nneg,replace=False)]]
    booster=lgb.LGBMClassifier(n_estimators=args.coarse_trees, learning_rate=.04,
                              num_leaves=15,max_depth=6,min_child_samples=80,
                              reg_lambda=5,random_state=args.seed,verbosity=-1,n_jobs=-1)
    booster.fit(coarse_fit[coarse_features],coarse_fit.label,categorical_feature=['category'])
    del coarse_fit
    print('Temporary LightGBM fitted ONLY on period A',flush=True)
    # Use SAME snapshot and held-out B/C labels, not A. Avoid adapting B/C features with future clicks.
    # Restrict candidate set against A/B earlier positives (at the corresponding boundary).
    allbase=base.drop(columns=['label'],errors='ignore')
    allbase['category']=pd.Categorical(allbase.category,categories=cats)
    scored=add_coarse_scores(allbase,booster,coarse_features,args.coarse_trees,cats,args.topk)
    # Current B/C are scored from the pre-A snapshot, so coarse has seen no B/C label.
    # Feature history is the same as the candidate snapshot to avoid using labels in features.
    topics,topic_keys=topic_ids_for_notes(notes)
    cutoff=int(h.timestamp.max())
    fit_b,truth_b=finalize(scored,h,notes,cutoff,topics,b)
    val_c,truth_c=finalize(scored,h,notes,cutoff,topics,c)
    # B/C candidates are same frozen snapshot: evaluation of C removes B positives from GT
    # (and from candidates), but B labels never enter feature computation.
    b_pairs=pd.MultiIndex.from_frame(b[['user_id','note_id']].drop_duplicates())
    idx=pd.MultiIndex.from_frame(val_c[['user_id','note_id']])
    val_c=val_c.loc[~idx.isin(b_pairs)].copy().reset_index(drop=True)
    c_label_pairs=pd.MultiIndex.from_frame(c[['user_id','note_id']].drop_duplicates())
    c_label_pairs=c_label_pairs[~c_label_pairs.isin(b_pairs)]
    c_eligible=(notes.publish_ts.to_numpy() <= cutoff)
    c_truth=c[['user_id','note_id']].drop_duplicates()
    c_truth=c_truth.loc[c_eligible[c_truth.note_id.to_numpy('int64')]]
    c_truth=c_truth.loc[~pd.MultiIndex.from_frame(c_truth).isin(b_pairs)]
    truth_c=c_truth.groupby('user_id').note_id.agg(set).to_dict()
    if not len(fit_b) or not len(val_c) or not fit_b.label.sum() or not val_c.label.sum():
        raise ValueError('No DeepFM training/validation positives, inspect candidate reach')
    print(f'DeepFM temporal train B: {len(fit_b):,} candidates, {int(fit_b.label.sum()):,} positives')
    print(f'DeepFM temporal val C: {len(val_c):,} candidates, {int(val_c.label.sum()):,} positives')
    # Create actual valid/test features from respective cutoff snapshots.
    frames={};truths={}
    for name,history,target,cut in [('valid',train,valid,int(train.timestamp.max())),
                                     ('test',pd.concat([train,valid],ignore_index=True),test,int(valid.timestamp.max()))]:
        raw=read(args.coarse_dir,f'coarse_{name}_top200')
        # Recover all 5 channel features because coarse export stores only IDs and prediction.
        merged_stage=merge_channels(args.data,name)
        cols=['user_id','note_id','rrf_score','channel_count']+CHANNEL_FEATURES
        raw=raw.merge(merged_stage[cols],on=['user_id','note_id'],how='left',validate='one_to_one')
        if raw[CHANNEL_FEATURES].isna().any().any():
            raise ValueError(f'{name} coarse candidates missing retrieval feature rows')
        del merged_stage
        if 'prediction' not in raw: raise ValueError(f'{name} needs coarse prediction field')
        if 'coarse_rank' not in raw:
            raw=filter_topk(raw,'prediction',200)
        frame,truth=finalize(raw,history,notes,cut,topics,target)
        frames[name]=frame;truths[name]=truth
        print(f'{name}: {len(frame):,} candidates, {int(frame.label.sum()):,} positives',flush=True)
    # Fit on B only. Never add A/C/future or valid/test labels to training tensors.
    pos_idx=np.flatnonzero(fit_b.label.to_numpy()>0);neg_idx=np.flatnonzero(fit_b.label.to_numpy()==0)
    chosen=np.r_[pos_idx,rng.choice(neg_idx,size=min(len(neg_idx),len(pos_idx)*args.negative_ratio),replace=False)]
    fitting=fit_b.iloc[chosen].reset_index(drop=True)
    mean=fitting[DENSE].to_numpy('float32').mean(0)
    std=np.maximum(fitting[DENSE].to_numpy('float32').std(0),1e-6)
    sp,de,bas=make_tensors(fitting,mean,std)
    labels=torch.from_numpy(fitting.label.to_numpy('float32',copy=True))
    loader=DataLoader(TensorDataset(sp,de,bas,labels),batch_size=args.batch_size,shuffle=True,
                      pin_memory=(device.type=='cuda'))
    model=ResidualDeepFM([len(users),len(notes),len(topic_keys)],len(DENSE)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=5e-5)
    pos_weight=torch.tensor(min(float((labels==0).sum()/max(float((labels==1).sum()),1.)),20.),device=device)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    base_c=topk_metric(val_c,truth_c,'prediction',20)
    # Epoch 0 = pure coarse ranking, retained whenever residual degrades C NDCG.
    best=base_c[1];best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()});bad=0
    print(f'C baseline LightGBM Recall@20={base_c[0]:.6f} NDCG@20={base_c[1]:.6f}',flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();total=0.
        for x,z,q,y in loader:
            x=x.to(device);z=z.to(device);q=q.to(device);y=y.to(device)
            logit=model(x,z,q)
            loss=loss_fn(logit,y)
            opt.zero_grad();loss.backward();opt.step()
            total+=float(loss.detach())*len(y)
        vals=infer(model,val_c,mean,std,device,args.batch_size)
        temp=val_c[['user_id','note_id']].copy();temp['fine_score']=vals
        r,n=topk_metric(temp,truth_c,'fine_score',20)
        print(f'Epoch {epoch}: loss={total/len(fitting):.5f}, C Recall@20={r:.6f}, C NDCG@20={n:.6f}',flush=True)
        if n>best+1e-8:
            best=n;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()});bad=0
        else:
            bad+=1
            if bad>=args.patience:
                print('Early stopping',flush=True);break
    model.load_state_dict(best_state)
    path=args.output/'deepfm_residual_v6.pt'
    torch.save({'state_dict':best_state,'dims':[len(users),len(notes),len(topic_keys)],
                'dense':DENSE,'sparse':SPARSE,'mean':mean,'std':std,
                'topic_keys':topic_keys,'seed':args.seed},path)
    for name,frame in frames.items():
        scores=infer(model,frame,mean,std,device,args.batch_size)
        out=frame[['user_id','note_id','prediction','coarse_rank']].copy()
        out['fine_score']=scores
        out=(out.sort_values(['user_id','fine_score','note_id'],ascending=[True,False,True])
                .reset_index(drop=True))
        out['fine_rank']=out.groupby('user_id').cumcount().add(1)
        dest=args.output/f'fine_{name}_top200.parquet'
        try:
            out.to_parquet(dest,index=False)
        except ImportError:
            dest=dest.with_suffix('.csv')
            out.to_csv(dest,index=False)
        for k in (20,100,200):
            r,n=topk_metric(out,truths[name],'fine_score',k)
            br,bn=topk_metric(out,truths[name],'prediction',k)
            print(f'{name} K={k}: DeepFM Recall={r:.6f} NDCG={n:.6f} | LGBM Recall={br:.6f} NDCG={bn:.6f}',flush=True)
        print(f'Saved {dest}',flush=True)
    print(f'Checkpoint {path}')
    print('Protocol: temporal A(train coarse)/B(train fine)/C(select fine); validation/test exports are post-training diagnostics.')
    print('Caveat: final coarse model was selected using external valid; valid-stage fine metrics remain optimistic.')

if __name__=='__main__':main()
