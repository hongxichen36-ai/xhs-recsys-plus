from pathlib import Path
import numpy as np
import pandas as pd


DATA_DIR = Path("data/raw_v6")
TOPK = 200


def dump_recall(score_fn, split_name, train, target, notes,
                prefix, topk=TOPK, batch_size=256):
    """
    把 score_fn 在某个 split 上的 Top-K 召回明细导出为
    data/raw_v5/{prefix}_{split_name}.parquet

    score_fn: 接受 (B,) user_id 数组，返回 (B, n_items) float32 分数矩阵
    """
    train_sets  = train.groupby("user_id").note_id.agg(set).to_dict()
    target_sets = target.groupby("user_id").note_id.agg(set).to_dict()

    publish  = notes.publish_ts.to_numpy()
    cutoff   = int(target.timestamp.min())
    eligible = publish <= cutoff
    n_items  = len(notes)

    users_eval = np.array(sorted(target_sets), dtype=np.int64)

    rows = []
    for start in range(0, len(users_eval), batch_size):
        uu = users_eval[start:start + batch_size]
        S = score_fn(uu).astype(np.float32, copy=True)
        S[:, ~eligible] = -np.inf

        for r, u in enumerate(uu):
            u = int(u)
            seen = train_sets.get(u, set())
            if seen:
                S[r, list(seen)] = -np.inf

            kk = min(topk, int(np.isfinite(S[r]).sum()))
            if kk == 0:
                continue

            ids = np.argpartition(-S[r], kk - 1)[:kk]
            ids = ids[np.argsort(-S[r, ids])]

            for rank, nid in enumerate(ids, start=1):
                rows.append((u, int(nid), float(S[r, nid]), rank))

    df = pd.DataFrame(rows, columns=["user_id", "note_id", "score", "rank"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"{prefix}_{split_name}.parquet"
    df.to_parquet(out, index=False)
    print(f"已导出 {out}  ({len(df):,} 条, {df.user_id.nunique():,} 用户)")