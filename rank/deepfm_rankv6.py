"""v6 DeepFM reranker. Run: python rank/deepfm_v6.py
Training: merged_recall_train candidates / rank_train_future labels.
Validation/test: coarse_valid/test_top200 candidate lists; their cutoff historical features.
No coarse score or coarse rank enters DeepFM (avoids in-sample coarse-score leakage).
WARNING: upstream coarse model was selected on valid labels; valid metrics are selection-biased.
For formal experiments rebuild upstream coarse validation candidates via out-of-fold protocol.
"""
import argparse
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent

def read(root, stem):
    for ext in ('.parquet', '.csv'):
        p = root / (stem + ext)
        if p.exists():
            return pd.read_parquet(p) if ext == '.parquet' else pd.read_csv(p)
    raise FileNotFoundError(f'缺少 {root / stem}.parquet/.csv')

def features(candidates, history, notes, cutoff, topics):
    """Only historical features at cutoff, no future labels or latent true preferences."""
    if len(history) and int(history.timestamp.max()) > cutoff:
        raise ValueError('history exceeds cutoff')
    meta = notes[['note_id', 'category', 'publish_ts']].copy()
    meta['topic_id'] = topics
    df = candidates[['user_id', 'note_id']].drop_duplicates().copy()
    df = df.merge(meta, on='note_id', how='left', validate='many_to_one')
    if df.publish_ts.isna().any():
        raise ValueError('unknown note_id')
    df = df[df.publish_ts <= cutoff].copy()
    h = history[['user_id', 'note_id']].drop_duplicates().copy()
    seen = pd.MultiIndex.from_frame(h)
    pairs = pd.MultiIndex.from_frame(df[['user_id', 'note_id']])
    df = df.loc[~pairs.isin(seen)].copy()
    hm = h.merge(meta[['note_id', 'topic_id']], on='note_id', how='left')
    uc = h.groupby('user_id').size()
    ic = h.groupby('note_id').size()
    tc = hm.groupby(['user_id', 'topic_id']).size()
    df['user_hist'] = df.user_id.map(uc).fillna(0).to_numpy(np.float32)
    df['item_pop'] = df.note_id.map(ic).fillna(0).to_numpy(np.float32)
    key = pd.MultiIndex.from_frame(df[['user_id', 'topic_id']])
    df['user_topic_count'] = np.asarray(tc.reindex(key, fill_value=0), dtype=np.float32)
    df['topic_affinity'] = df.user_topic_count / np.maximum(df.user_hist, 1)
    df['age_days'] = (cutoff - df.publish_ts.to_numpy()) / 86400.
    df['user_hist'] = np.log1p(df['user_hist'])
    df['item_pop'] = np.log1p(df['item_pop'])
    return df[['user_id', 'note_id', 'topic_id', 'user_hist', 'item_pop', 'topic_affinity', 'age_days']].reset_index(drop=True)

def label_df(df, future, notes, cutoff, history):
    eligible = notes.set_index('note_id').publish_ts
    pos = future[['user_id','note_id']].drop_duplicates().copy()
    pos = pos[pos.note_id.map(eligible) <= cutoff]
    seen = pd.MultiIndex.from_frame(history[['user_id','note_id']].drop_duplicates())
    pos = pos.loc[~pd.MultiIndex.from_frame(pos).isin(seen)].copy()
    pos_keys = pd.MultiIndex.from_frame(pos)
    df['label'] = pd.MultiIndex.from_frame(df[['user_id','note_id']]).isin(pos_keys).astype(np.float32)
    return df, pos.groupby('user_id').note_id.agg(set).to_dict()

SPARSE = ['user_id','note_id','topic_id']
DENSE = ['user_hist','item_pop','topic_affinity','age_days']

class DeepFM(nn.Module):
    def __init__(self, dims, dense_dim=4, emb_dim=16):
        super().__init__()
        self.linear = nn.ModuleList([nn.Embedding(n, 1) for n in dims])
        self.emb = nn.ModuleList([nn.Embedding(n, emb_dim) for n in dims])
        self.dense_linear = nn.Linear(dense_dim, 1)
        self.mlp = nn.Sequential(nn.Linear(len(dims)*emb_dim+dense_dim,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        for e in self.emb: nn.init.normal_(e.weight, std=.02)
        for e in self.linear: nn.init.zeros_(e.weight)
    def forward(self, cats, nums):
        em = torch.stack([m(cats[:,j]) for j,m in enumerate(self.emb)], dim=1)
        linear = sum(m(cats[:,j]) for j,m in enumerate(self.linear)) + self.dense_linear(nums)
        summed = em.sum(1)
        fm = .5*(summed.square()-em.square().sum(1)).sum(1,keepdim=True)
        deep = self.mlp(torch.cat([em.flatten(1),nums],dim=1))
        return (linear+fm+deep).squeeze(1)

def tensors(df, mean, std):
    x = torch.from_numpy(df[SPARSE].to_numpy(np.int64,copy=True))
    a = df[DENSE].to_numpy(np.float32,copy=True)
    a = np.nan_to_num((a-mean)/std,nan=0.,posinf=0.,neginf=0.)
    z = torch.from_numpy(a)
    return x,z

@torch.no_grad()
def predict(model, df, mean, std, device, batch_size):
    model.eval()
    xs, zs = tensors(df,mean,std)
    results = []
    for start in range(0,len(df),batch_size):
        x = xs[start:start+batch_size].to(device)
        z = zs[start:start+batch_size].to(device)
        results.append(torch.sigmoid(model(x,z)).cpu().numpy())
    return np.concatenate(results) if results else np.zeros(0,np.float32)

def metric(df, truth, col='score', ks=(20,100,200)):
    sort = df.sort_values(['user_id',col,'note_id'],ascending=[True,False,True])
    grouped = sort.groupby('user_id',sort=False)
    ans = {}
    for k in ks:
        top = grouped.head(k).groupby('user_id').note_id.agg(list).to_dict()
        rr=[]; nn=[]
        for u,gt in truth.items():
            items=top.get(u,[])
            hits=np.fromiter((int(i in gt) for i in items),dtype=np.float64)
            rr.append(float(hits.sum()/len(gt)))
            discounts=1/np.log2(np.arange(len(hits))+2)
            ideal=(1/np.log2(np.arange(min(k,len(gt)))+2)).sum()
            nn.append(float(np.dot(hits,discounts)/ideal))
        ans[k]=(float(np.mean(rr)) if rr else 0., float(np.mean(nn)) if nn else 0.)
    return ans

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=ROOT/'data'/'raw_v6')
    p.add_argument('--coarse-dir',type=Path,default=ROOT/'rank'/'output_v6')
    p.add_argument('--out',type=Path,default=ROOT/'rank'/'output_v6')
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=4096)
    p.add_argument('--negative-ratio',type=int,default=15)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--patience',type=int,default=20)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    notes=read(args.data,'notes'); users=read(args.data,'users')
    history=read(args.data,'rank_train_history'); future=read(args.data,'rank_train_future')
    train=read(args.data,'train'); valid=read(args.data,'valid'); test=read(args.data,'test')
    if not np.array_equal(notes.note_id.to_numpy(),np.arange(len(notes))):raise ValueError('note ids must be contiguous')
    if not np.array_equal(users.user_id.to_numpy(),np.arange(len(users))):raise ValueError('user ids must be contiguous')
    # topics: use explicitly available note tags, not true user interest or micro_group.
    # category + first displayed tag prevents identical named tags in different categories from collapsing.
    first_tag=notes.tags.fillna('').astype(str).str.split(',').str[0] if 'tags' in notes else notes.category.astype(str)
    note_topic=notes.category.astype(str)+'::'+first_tag
    topic_ids,_=pd.factorize(note_topic, sort=True)
    topic_ids=topic_ids.astype(np.int64)
    dims=[len(users),len(notes),int(topic_ids.max())+1]
    frames={}; truths={}
    stages=[('train', read(args.data,'merged_recall_train'), history, future),
            ('valid', read(args.coarse_dir,'coarse_valid_top200'),train,valid),
            ('test', read(args.coarse_dir,'coarse_test_top200'),pd.concat([train,valid],ignore_index=True),test)]
    for name,candidates,h,f in stages:
        cutoff=int(h.timestamp.max())
        df=features(candidates,h,notes,cutoff,topic_ids)
        df,truth=label_df(df,f,notes,cutoff,h)
        frames[name]=df;truths[name]=truth
        print(f'{name}: candidates={len(df):,} positives={int(df.label.sum()):,} truth_users={len(truth):,}',flush=True)
    tr,va,te=frames['train'],frames['valid'],frames['test']
    if tr.label.sum()==0 or va.label.sum()==0:raise RuntimeError('No positives in training/validation candidate sets')
    # Sample negatives ONLY in fit; validation/test retain all candidates.
    pos_idx=np.flatnonzero(tr.label.to_numpy()>0)
    neg_idx=np.flatnonzero(tr.label.to_numpy()==0)
    count=min(len(neg_idx),len(pos_idx)*args.negative_ratio)
    chosen=np.concatenate([pos_idx,rng.choice(neg_idx,count,replace=False)])
    fit=tr.iloc[chosen].reset_index(drop=True)
    mean=fit[DENSE].to_numpy(np.float32).mean(0)
    std=fit[DENSE].to_numpy(np.float32).std(0);std=np.maximum(std,1e-6)
    xs,zs=tensors(fit,mean,std)
    ys=torch.from_numpy(fit.label.to_numpy(np.float32))
    loader=DataLoader(TensorDataset(xs,zs,ys),batch_size=args.batch_size,shuffle=True,pin_memory=(device.type=='cuda'))
    model=DeepFM(dims).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    pos_weight=torch.tensor(min(float((ys==0).sum()/max(float((ys==1).sum()),1.)),40.),device=device)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best=-1.;best_weights=None;bad=0
    for epoch in range(1,args.epochs+1):
        model.train();total=0.
        for x,z,y in loader:
            x=x.to(device);z=z.to(device);y=y.to(device)
            loss=loss_fn(model(x,z),y)
            opt.zero_grad();loss.backward();opt.step()
            total+=float(loss.detach())*len(y)
        scores=predict(model,va,mean,std,device,args.batch_size)
        tmp=va[['user_id','note_id']].copy();tmp['score']=scores
        met=metric(tmp,truths['valid'],ks=(20,200))
        print(f'Epoch {epoch} loss={total/len(fit):.5f} Valid Recall@20={met[20][0]:.5f} NDCG@20={met[20][1]:.5f} Recall@200={met[200][0]:.5f}',flush=True)
        if met[20][1] > best + 1e-8:
            best=met[20][1];best_weights=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()});bad=0
        else:
            bad+=1
            if bad>=args.patience:
                print('Early stop',flush=True);break
    model.load_state_dict(best_weights)
    checkpoint=args.out/'deepfm_v6.pt'
    torch.save({'model':best_weights,'dims':dims,'dense_mean':mean,'dense_std':std,'sparse_cols':SPARSE,'dense_cols':DENSE,'topic_keys':sorted(note_topic.unique()),'seed':args.seed},checkpoint)
    for name in ('valid','test'):
        df=frames[name];scores=predict(model,df,mean,std,device,args.batch_size)
        out=df[['user_id','note_id']].copy();out['fine_score']=scores
        out=out.sort_values(['user_id','fine_score','note_id'],ascending=[True,False,True])
        out['fine_rank']=out.groupby('user_id').cumcount()+1
        filepath=args.out/f'fine_{name}_top200.parquet'
        try:
            out.to_parquet(filepath,index=False)
        except ImportError:
            filepath=filepath.with_suffix('.csv')
            out.to_csv(filepath,index=False)
        print(f'{name}: {metric(out.rename(columns={"fine_score":"score"}),truths[name],ks=(20,100,200))} -> {filepath}',flush=True)
    print(f'Checkpoint: {checkpoint}',flush=True)
    print('CAVEAT: upstream coarse model was tuned using valid labels; this is an engineering baseline, not leakage-free final evaluation.')

if __name__=='__main__': main()
