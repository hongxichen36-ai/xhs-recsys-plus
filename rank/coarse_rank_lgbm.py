"""V6 multi-channel LightGBM coarse ranking.
Requires per-channel *_train/valid/test recall files. The *_train files must be
produced by models fitted ONLY on rank_train_history (no rank_train_future).
Run: python rank/coarse_rank_v6.py
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=ROOT/'data'/'raw_v6')
    ap.add_argument('--output',type=Path,default=ROOT/'rank'/'output_v6')
    ap.add_argument('--neg-ratio',type=int,default=20)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    root=args.data; out=args.output
    check_inputs(root)
    out.mkdir(parents=True,exist_ok=True)
    notes=read(root,'notes'); train=read(root,'train'); valid=read(root,'valid'); test=read(root,'test')
    prefix=read(root,'rank_train_history'); future=read(root,'rank_train_future')
    if prefix.timestamp.max()>=future.timestamp.min(): raise ValueError('粗排训练历史和未来标签时间重叠')
    stages={'train':(prefix,future,int(prefix.timestamp.max())),
            'valid':(train,valid,int(train.timestamp.max())),
            'test':(pd.concat([train,valid],ignore_index=True),test,int(valid.timestamp.max()))}
    data={}; truths={}
    for split,(hist,target,cutoff) in stages.items():
        merged=merge_channels(root,split)
        frame,truth=make_samples(merged,hist,target,notes,cutoff)
        if frame.empty: raise ValueError(f'{split} 没有候选')
        data[split]=frame; truths[split]=truth
        print(f'{split}: candidates={len(frame):,}, labels={int(frame.label.sum()):,}, truth={len(truth):,}',flush=True)
        del merged
    exclude={'user_id','note_id','label','publish_ts'}
    features=[c for c in data['train'].columns if c not in exclude and c!='prediction']
    categories=sorted(notes.category.astype(str).unique())
    for d in data.values(): d['category']=pd.Categorical(d.category,categories=categories)
    tr=data['train']; va=data['valid']; te=data['test']
    if tr.label.sum()==0 or va.label.sum()==0: raise ValueError('训练/验证没有正标签；检查时间切分和召回候选')
    rng=np.random.default_rng(args.seed)
    pos=np.flatnonzero(tr.label.to_numpy()==1); neg=np.flatnonzero(tr.label.to_numpy()==0)
    nneg=min(len(neg),len(pos)*args.neg_ratio)
    sampled=np.r_[pos,rng.choice(neg,nneg,replace=False)]
    fit=tr.iloc[sampled]
    model=lgb.LGBMClassifier(n_estimators=160,learning_rate=.04,num_leaves=15,
        max_depth=6,min_child_samples=80,reg_lambda=5,random_state=args.seed,verbosity=-1,n_jobs=-1)
    model.fit(fit[features],fit.label,categorical_feature=['category'])
    # Select iteration based on ranking NDCG, not binary logloss
    best_iteration=1; best=-1
    for iteration in (1,10,20,40,60,80,120,160):
        va['prediction']=model.predict_proba(va[features],num_iteration=iteration)[:,1]
        r,n=metrics(va,truths['valid'],'prediction',200)
        print(f'iter={iteration}: valid Recall@200={r:.6f}, NDCG@200={n:.6f}',flush=True)
        if n>best: best=n;best_iteration=iteration
    print('Best iteration:',best_iteration)
    # ==========================================
    # 修复 Windows 中文路径保存问题
    # ==========================================

    model_path = out / "lightgbm_coarse_v6.txt"

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 在内存中导出 LightGBM 模型文本
    model_text = model.booster_.model_to_string(
        num_iteration=best_iteration
    )

    # 使用 Python UTF-8 文件接口保存
    model_path.write_text(
        model_text,
        encoding="utf-8"
    )

    print("LightGBM 模型保存成功:", model_path)
    for split,df in (('valid',va),('test',te)):
        df['prediction']=model.predict_proba(df[features],num_iteration=best_iteration)[:,1]
        for k in TOPKS:
            lr,ln=metrics(df,truths[split],'prediction',k)
            rr,rn=metrics(df,truths[split],'rrf_score',k)
            print(f'{split} K={k} LightGBM Recall={lr:.6f} NDCG={ln:.6f} | RRF Recall={rr:.6f} NDCG={rn:.6f}',flush=True)
        top=df.sort_values(['user_id','prediction','note_id'],ascending=[True,False,True]).groupby('user_id',sort=False).head(200).copy()
        top['coarse_rank']=top.groupby('user_id').cumcount()+1
        top[['user_id','note_id','prediction','coarse_rank']].to_parquet(out/f'coarse_{split}_top200.parquet',index=False)
    pd.DataFrame({'feature':features,'importance':model.feature_importances_}).sort_values('importance',ascending=False).to_csv(out/'feature_importance.csv',index=False)
    print('Saved:',out)

if __name__=='__main__':main()
