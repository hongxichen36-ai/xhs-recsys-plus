"""Export five leakage-controlled retrieval channels for v6 coarse-ranker training.
Only rank_train_history is used. rank_train_future is NEVER read.
Run: python recall/export_five_train_v6.py
Dependencies: numpy pandas scipy torch pyarrow
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
# When copied into recall/, ROOT points to project root.
CHANNELS = ('mostpopular_decay', 'itemcf_weighted', 'bprmf', 'lightgcn', 'twotower')
DAY = 86400.0

def read(root, stem):
    for suffix in ('.parquet', '.csv'):
        path = root / (stem + suffix)
        if path.exists():
            return pd.read_parquet(path) if suffix == '.parquet' else pd.read_csv(path)
    raise FileNotFoundError(f'Missing {stem} under {root}')

def preference(history, like, collect, comment):
    weights = np.ones(len(history), dtype=np.float32)
    for col, factor in (('liked', like), ('collected', collect), ('commented', comment)):
        if col not in history.columns:
            raise ValueError(f'History missing {col}')
        vals = history[col].to_numpy(dtype=np.float32)
        if not np.isin(vals, [0, 1]).all():
            raise ValueError(f'{col} is not a binary field')
        weights += factor * vals
    return weights

def make_csr(u, i, w, nu, ni):
    X = csr_matrix((np.asarray(w, np.float32), (u, i)), shape=(nu, ni))
    X.sum_duplicates()
    X.sort_indices()
    return X

def export_topk(channel, score_fn, users, ni, published, cutoff, seen, output, topk, batch):
    eligible = published <= cutoff
    chunks = []
    for start in range(0, len(users), batch):
        uu = users[start:start+batch]
        scores = np.array(score_fn(uu), dtype=np.float32, copy=True)
        if scores.shape != (len(uu), ni):
            raise ValueError(f'{channel}: score shape {scores.shape}')
        scores[:, ~eligible] = -np.inf
        for row, uid in enumerate(uu):
            a, b = seen.indptr[uid], seen.indptr[uid+1]
            scores[row, seen.indices[a:b]] = -np.inf
        for row, uid in enumerate(uu):
            available = np.flatnonzero(np.isfinite(scores[row]))
            k = min(topk, len(available))
            if not k:
                continue
            pick = available[np.argpartition(-scores[row, available], k-1)[:k]]
            pick = pick[np.lexsort((pick, -scores[row, pick]))]
            chunks.append(pd.DataFrame({
                'user_id': np.full(k, uid, dtype=np.int32),
                'note_id': pick.astype(np.int32),
                'score': scores[row, pick],
                'rank': np.arange(1, k+1, dtype=np.int16),
            }))
    if not chunks:
        raise ValueError(f'{channel}: no candidates')
    out = pd.concat(chunks, ignore_index=True)
    if out.duplicated(['user_id', 'note_id']).any():
        raise AssertionError(f'{channel}: duplicate user-item')
    if (published[out.note_id.to_numpy()] > cutoff).any():
        raise AssertionError(f'{channel}: unpublished notes in candidates')
    keys = out.user_id.to_numpy(np.int64)*ni + out.note_id.to_numpy(np.int64)
    if np.isin(keys, np.sort(seen.nonzero()[0].astype(np.int64)*ni+seen.nonzero()[1])).any():
        raise AssertionError(f'{channel}: historically clicked notes in candidates')
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.to_parquet(output, index=False)
    except ImportError:
        output = output.with_suffix('.csv')
        out.to_csv(output, index=False)
    print(f'{channel}: {len(out):,} rows / {out.user_id.nunique():,} users -> {output}', flush=True)

def popularity(history, notes, cutoff, weights, args):
    ni = len(notes)
    ages = (cutoff - history.timestamp.to_numpy(np.int64))/DAY
    decayed = weights * np.exp(-np.log(2)*ages/args.action_half_life)
    heat = np.bincount(history.note_id.to_numpy(np.int64), weights=decayed, minlength=ni)
    age_note = np.maximum(0, (cutoff-notes.publish_ts.to_numpy(np.int64))/DAY)
    freshness = np.exp(-np.log(2)*age_note/args.publish_half_life)
    s = (np.log1p(heat) + args.freshness_weight * freshness).astype(np.float32)
    return lambda users: np.broadcast_to(s, (len(users), ni))

def weighted_itemcf(history, seen, weights, nu, ni, args):
    limited = history.assign(_w=weights).sort_values('timestamp').groupby('user_id').tail(args.max_history)
    X = make_csr(limited.user_id.to_numpy(), limited.note_id.to_numpy(), limited._w.to_numpy(), nu, ni)
    norms = np.sqrt(np.maximum(np.asarray(X.power(2).sum(axis=0)).ravel(), 1e-12))
    sim = (X.T @ X).tocsr()
    sim.setdiag(0)
    sim.eliminate_zeros()
    inv = 1 / norms
    sim = sim.multiply(inv[:, None]).multiply(inv[None, :]).tocsr()
    # Keep at most args.neighbors outgoing edges per item (sparse CSR rows).
    indptr=[0]; ix=[]; vals=[]
    for j in range(ni):
        a,b=sim.indptr[j:j+2]
        c=sim.indices[a:b]; v=sim.data[a:b]
        if len(v)>args.neighbors:
            keep=np.argpartition(v,-args.neighbors)[-args.neighbors:]
            c=c[keep];v=v[keep]
        ix.extend(c);vals.extend(v);indptr.append(len(ix))
    sim = csr_matrix((np.asarray(vals,np.float32), np.asarray(ix,np.int32), np.asarray(indptr,np.int64)),shape=(ni,ni))
    pref = make_csr(history.user_id.to_numpy(), history.note_id.to_numpy(), weights, nu, ni)
    pop = np.bincount(history.note_id.to_numpy(np.int64), minlength=ni).astype(np.float32)
    pop /= max(pop.max(), 1.0)
    print(f'ItemCF similarity edges={sim.nnz:,}, max neighbors={np.diff(sim.indptr).max()}', flush=True)
    return lambda users: (pref[users] @ sim).toarray() + 1e-5*pop[None,:]

class MF(nn.Module):
    def __init__(self, nu,ni,d):
        super().__init__()
        self.user=nn.Embedding(nu,d)
        self.item=nn.Embedding(ni,d)
        nn.init.normal_(self.user.weight,std=.05)
        nn.init.normal_(self.item.weight,std=.05)

class GCN(MF):
    def __init__(self,nu,ni,d,layers,u,i):
        super().__init__(nu,ni,d)
        self.layers=layers
        src=np.r_[u,i+nu]; dst=np.r_[i+nu,u]
        degree=np.bincount(src,minlength=nu+ni).astype(np.float32)
        val=(np.maximum(degree[src],1)*np.maximum(degree[dst],1))**-.5
        adj=torch.sparse_coo_tensor(torch.from_numpy(np.stack([src,dst])).long(),torch.from_numpy(val.astype(np.float32)),(nu+ni,nu+ni)).coalesce()
        self.register_buffer('adj',adj)
    def embeddings(self):
        z=torch.cat([self.user.weight,self.item.weight])
        total=z
        for _ in range(self.layers):
            z=torch.sparse.mm(self.adj,z)
            total=total+z
        total=total/(self.layers+1)
        return total[:self.user.num_embeddings],total[self.user.num_embeddings:]

class Tower(nn.Module):
    def __init__(self,nu,ni,nc,d,u_cnt,i_cnt,profile):
        super().__init__()
        self.uid=nn.Embedding(nu,d);self.iid=nn.Embedding(ni,d);self.cat=nn.Embedding(nc,d)
        self.fc=nn.Linear(nc,d,bias=False)
        for emb in (self.uid,self.iid,self.cat):
            nn.init.normal_(emb.weight,std=.05)
        self.register_buffer('ug',torch.as_tensor(u_cnt/(u_cnt+5),dtype=torch.float32))
        self.register_buffer('ig',torch.as_tensor(i_cnt/(i_cnt+10),dtype=torch.float32))
        self.register_buffer('profile',torch.as_tensor(profile,dtype=torch.float32))
    def uv(self,u):
        return F.normalize(self.uid(u)*self.ug[u,None]+self.fc(self.profile[u]),dim=-1)
    def iv(self,i,categories):
        return F.normalize(self.iid(i)*self.ig[i,None]+self.cat(categories[i]),dim=-1)
def sample_negative_by_event_time(
    users,
    positive_times,
    published,
    positive_keys,
    n_items,
    rng,
    max_attempts=30
):
    """
    基于每条正反馈事件时间的负采样。

    要求：
    1. publish_ts[j] <= positive_times
    2. (user_id, j) 不在训练历史正反馈中
    3. 不使用未来点击标签

    返回：
        negatives: 负样本笔记 ID
        valid_mask: 哪些正反馈有合法负样本
    """

    users = np.asarray(users, dtype=np.int64)
    positive_times = np.asarray(
        positive_times, dtype=np.int64
    )

    # 笔记按发布时间排序
    sorted_items = np.argsort(
        published, kind="stable"
    )

    sorted_times = published[sorted_items]

    # 每个正反馈事件发生时，
    # 已经发布的笔记数量
    available_counts = np.searchsorted(
        sorted_times,
        positive_times,
        side="right"
    )

    n = len(users)

    negatives = np.full(
        n, -1, dtype=np.int64
    )

    valid_mask = available_counts > 0

    unresolved = np.flatnonzero(valid_mask)

    # ---------------------------------------------
    # 第一阶段：批量拒绝采样
    # ---------------------------------------------

    for _ in range(max_attempts):

        if len(unresolved) == 0:
            break

        # 每个事件只从当时已发布的笔记中抽样
        counts = available_counts[unresolved]

        random_positions = (
            rng.random(len(unresolved)) * counts
        ).astype(np.int64)

        candidates = sorted_items[random_positions]

        keys = (
            users[unresolved] * n_items
            + candidates
        )

        positions = np.searchsorted(
            positive_keys, keys
        )

        # 安全索引，防止越界
        safe_positions = np.minimum(
            positions,
            len(positive_keys) - 1
        )

        # 是否属于训练历史正反馈
        is_positive = (
            (positions < len(positive_keys))
            & (
                positive_keys[safe_positions]
                == keys
            )
        )

        accepted = ~is_positive

        negatives[unresolved[accepted]] = (
            candidates[accepted]
        )

        unresolved = unresolved[is_positive]

    # ---------------------------------------------
    # 第二阶段：精确回退
    # 处理剩余未抽到合法负样本的事件
    # ---------------------------------------------

    for idx in unresolved:

        uid = int(users[idx])
        count = int(available_counts[idx])

        available_items = sorted_items[:count]

        candidate_keys = (
            uid * n_items + available_items
        )

        positions = np.searchsorted(
            positive_keys, candidate_keys
        )

        safe_positions = np.minimum(
            positions,
            len(positive_keys) - 1
        )

        is_positive = (
            (positions < len(positive_keys))
            & (
                positive_keys[safe_positions]
                == candidate_keys
            )
        )

        eligible_negatives = available_items[
            ~is_positive
        ]

        if len(eligible_negatives) == 0:
            # 当前事件不存在合法负样本
            valid_mask[idx] = False
        else:
            negatives[idx] = rng.choice(
                eligible_negatives
            )

    valid_mask &= negatives >= 0

    # ---------------------------------------------
    # 质量检查
    # ---------------------------------------------

    if valid_mask.any():

        jj = negatives[valid_mask]
        uu = users[valid_mask]
        tt = positive_times[valid_mask]

        assert np.all(published[jj] <= tt)

        neg_keys = uu * n_items + jj

        assert not np.isin(
            neg_keys, positive_keys
        ).any()

    return negatives, valid_mask
def train_embed(name,model,u,i,ts,published,positive_keys,nu,ni,args,device,cats=None):
    model=model.to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    rng=np.random.default_rng(args.seed+17)
    ct=torch.as_tensor(cats,dtype=torch.long,device=device) if cats is not None else None
    epochs=args.tower_epochs if name=='twotower' else args.epochs
    # Negatives: not among prefix positives and published at the sampled positive's time.
    # A draw is still an unobserved interaction, not a certified dislike.
    for epoch in range(epochs):

        model.train()

        order = rng.permutation(len(u))

        loss_total = 0.0
        trained_count = 0
        skipped = 0

        for start in range(0, len(u), args.batch_size):

            ix = order[start:start + args.batch_size]

            uu = u[ix]
            ii = i[ix]

            # ==========================================
            # 1. 事件时间负采样
            # ==========================================

            jj, valid_mask = sample_negative_by_event_time(
                users=uu,
                positive_times=ts[ix],
                published=published,
                positive_keys=positive_keys,
                n_items=ni,
                rng=rng
            )

            # 没有合法负样本时跳过
            if not valid_mask.any():
                skipped += len(ix)
                continue

            skipped += int((~valid_mask).sum())

            # 过滤无法采样的正样本
            uu = uu[valid_mask]
            ii = ii[valid_mask]
            jj = jj[valid_mask]

            # ==========================================
            # 2. 转换成 PyTorch Tensor
            # ==========================================

            tu = torch.as_tensor(
                uu,
                dtype=torch.long,
                device=device
            )

            ti = torch.as_tensor(
                ii,
                dtype=torch.long,
                device=device
            )

            tj = torch.as_tensor(
                jj,
                dtype=torch.long,
                device=device
            )

            # ==========================================
            # 3. 各模型的向量表示
            # ==========================================

            if name == 'lightgcn':

                eu, ei = model.embeddings()

                pu = eu[tu]
                pi = ei[ti]
                pj = ei[tj]

                reg = (
                    model.user(tu).square().sum(-1)
                    + model.item(ti).square().sum(-1)
                    + model.item(tj).square().sum(-1)
                ).mean() / 2

                scale = 1

            elif name == 'twotower':

                pu = model.uv(tu)

                pi = model.iv(ti, ct)
                pj = model.iv(tj, ct)

                reg = (
                    model.uid(tu).square().sum(-1)
                    + model.iid(ti).square().sum(-1)
                    + model.iid(tj).square().sum(-1)
                ).mean() / 2

                scale = 10

            else:

                pu = model.user(tu)
                pi = model.item(ti)
                pj = model.item(tj)

                reg = (
                    pu.square().sum(-1)
                    + pi.square().sum(-1)
                    + pj.square().sum(-1)
                ).mean() / 2

                scale = 1

            # ==========================================
            # 4. BPR Loss
            # ==========================================

            pos_score = (pu * pi).sum(-1)
            neg_score = (pu * pj).sum(-1)

            loss = (
                F.softplus(
                    (neg_score - pos_score) * scale
                ).mean()
                + args.reg * reg
            )

            # ==========================================
            # 5. 反向传播
            # ==========================================

            opt.zero_grad()
            loss.backward()
            opt.step()

            # 这里必须使用过滤后的样本数量
            batch_count = len(uu)

            loss_total += loss.item() * batch_count
            trained_count += batch_count

        # ==========================================
        # 6. Epoch 统计
        # ==========================================

        if trained_count == 0:
            raise RuntimeError(
                "本轮没有任何合法训练样本，"
                "请检查发布时间与负采样约束"
            )

        print(
            f"{name} epoch {epoch+1}/{epochs}: "
            f"loss={loss_total/trained_count:.5f}, "
            f"trained={trained_count:,}, "
            f"skipped={skipped:,}"
        )
    model.eval()
    with torch.no_grad():
        if name=='lightgcn': eu,ei=model.embeddings()
        elif name=='twotower':
            eu=torch.cat([model.uv(x) for x in torch.arange(nu,device=device).split(1024)])
            ei=torch.cat([model.iv(x,ct) for x in torch.arange(ni,device=device).split(4096)])
        else:eu,ei=model.user.weight,model.item.weight
        eu=eu.detach();ei=ei.detach()
    def score(users):
        with torch.no_grad():
            out=eu[torch.as_tensor(users,device=device)]@ei.T
            return out.cpu().numpy()
    return score

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=ROOT/'data'/'raw_v6')
    p.add_argument('--model',choices=CHANNELS+('all',),default='all')
    p.add_argument('--topk',type=int,default=200)
    p.add_argument('--batch',type=int,default=48)
    p.add_argument('--neighbors',type=int,default=200)
    p.add_argument('--max-history',type=int,default=100)
    p.add_argument('--action-half-life',type=float,default=3.0)
    p.add_argument('--publish-half-life',type=float,default=7.0)
    p.add_argument('--freshness-weight',type=float,default=.15)
    p.add_argument('--like-weight',type=float,default=1.0)
    p.add_argument('--collect-weight',type=float,default=1.0)
    p.add_argument('--comment-weight',type=float,default=1.0)
    p.add_argument('--pop-like-weight',type=float,default=.5)
    p.add_argument('--pop-collect-weight',type=float,default=1.0)
    p.add_argument('--pop-comment-weight',type=float,default=1.5)
    p.add_argument('--dim',type=int,default=64)
    p.add_argument('--layers',type=int,default=3)
    p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--tower-epochs',type=int,default=25)
    p.add_argument('--batch-size',type=int,default=4096)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--reg',type=float,default=.0001)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if min(args.topk,args.batch,args.neighbors,args.max_history,args.batch_size,args.dim) <= 0:
        raise ValueError('All size parameters must be positive')
    if min(args.action_half_life,args.publish_half_life)<=0:
        raise ValueError('Half-lives must be positive')
    np.random.seed(args.seed);torch.manual_seed(args.seed)
    root=args.data
    users=read(root,'users');notes=read(root,'notes');hist=read(root,'rank_train_history')
    nu,ni=len(users),len(notes)
    if not np.array_equal(users.user_id.to_numpy(),np.arange(nu)) or not np.array_equal(notes.note_id.to_numpy(),np.arange(ni)):
        raise ValueError('User and item IDs must be 0-based contiguous and sorted')
    if hist.empty or hist.duplicated(['user_id','note_id']).any():
        raise ValueError('History empty or duplicate pairs')
    u=hist.user_id.to_numpy(np.int64);i=hist.note_id.to_numpy(np.int64);ts=hist.timestamp.to_numpy(np.int64)
    published=notes.publish_ts.to_numpy(np.int64)
    cutoff=int(ts.max())
    if (ts<published[i]).any() or (ts>cutoff).any():raise ValueError('Invalid history timestamps')
    seen=make_csr(u,i,np.ones(len(u),np.float32),nu,ni)
    pos_keys=np.sort(np.unique(u*ni+i))
    all_users=np.arange(nu,dtype=np.int64)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'v6 history={len(hist):,}, users={nu:,}, notes={ni:,}, cutoff={cutoff}, device={device}',flush=True)
    for name in CHANNELS if args.model=='all' else [args.model]:
        print(f'\n=== {name} ===',flush=True)
        if name=='mostpopular_decay':
            w=preference(hist,args.pop_like_weight,args.pop_collect_weight,args.pop_comment_weight)
            scorer=popularity(hist,notes,cutoff,w,args)
        elif name=='itemcf_weighted':
            w=preference(hist,args.like_weight,args.collect_weight,args.comment_weight)
            scorer=weighted_itemcf(hist,seen,w,nu,ni,args)
        else:
            if name=='bprmf':model=MF(nu,ni,args.dim);cats=None
            elif name=='lightgcn':model=GCN(nu,ni,args.dim,args.layers,u,i);cats=None
            else:
                cats,_=pd.factorize(notes.category)
                nc=int(cats.max())+1
                profile=np.zeros((nu,nc),np.float32)
                np.add.at(profile,(u,cats[i]),1)
                profile/=np.maximum(profile.sum(axis=1,keepdims=True),1)
                uc=np.bincount(u,minlength=nu).astype(np.float32)
                ic=np.bincount(i,minlength=ni).astype(np.float32)
                model=Tower(nu,ni,nc,args.dim,uc,ic,profile)
            scorer=train_embed(name,model,u,i,ts,published,pos_keys,nu,ni,args,device,cats)
        export_topk(name,scorer,all_users,ni,published,cutoff,seen,root/f'{name}_train.parquet',args.topk,args.batch)
    print('\nFinished. All train recalls generated without future labels.',flush=True)

if __name__=='__main__':
    main()
