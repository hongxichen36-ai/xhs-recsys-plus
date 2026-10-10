from pathlib import Path

import numpy as np
import pandas as pd


def read_table(root, name):
    root = Path(root)
    p = root / (name + ".parquet")

    if p.exists():
        return pd.read_parquet(p)

    return pd.read_csv(root / (name + ".csv"))


def load_data(root):
    names = [
        "users", "notes", "train", "valid", "test"
    ]

    return [
        read_table(root, name)
        for name in names
    ]


def evaluation_data(
    train, valid, test, notes, split="valid"
):
    # 验证期使用训练交互作为已见历史
    # 测试期使用训练+验证交互作为已见历史
    if split == "valid":
        history = train
        target = valid
    else:
        history = pd.concat(
            [train, valid], ignore_index=True
        )
        target = test

    # 固定评价区间结束时间
    end_time = (
        int(target.timestamp.max())
        if len(target) else -1
    )

    eligible = (
        notes.publish_ts.to_numpy() <= end_time
    )

    seen = (
        history.groupby("user_id")
        .note_id.agg(set).to_dict()
    )

    truth = (
        target.groupby("user_id")
        .note_id.agg(set).to_dict()
    )

    return seen, truth, eligible


def evaluate(
    score_fn,
    train,
    valid,
    test,
    notes,
    split="valid",
    k=20,
    batch_size=64
):
    """
    全量候选 Top-K 评价。

    计算 Recall@K 与 NDCG@K。
    过滤已经交互的笔记。
    候选笔记必须在评价区间结束前发布。
    """

    seen, truth, eligible = evaluation_data(
        train, valid, test, notes, split
    )

    users = sorted(truth)

    recall = []
    ndcg = []

    for b in range(0, len(users), batch_size):
        us = users[b:b + batch_size]

        scores = np.array(
            score_fn(np.asarray(us, dtype=np.int64)),
            dtype=np.float32,
            copy=True
        )

        if scores.shape != (len(us), len(notes)):
            raise ValueError(
                f"Invalid score shape {scores.shape}"
            )

        scores[:, ~eligible] = -np.inf

        for row, u in enumerate(us):
            positive = truth[u] - seen.get(u, set())

            positive = {
                i for i in positive if eligible[i]
            }

            if not positive:
                continue

            if u in seen:
                scores[row, list(seen[u])] = -np.inf

            count = min(
                k,
                int(np.isfinite(scores[row]).sum())
            )

            if count == 0:
                continue

            ids = np.argpartition(
                -scores[row], count - 1
            )[:count]

            ids = ids[
                np.argsort(-scores[row, ids])
            ]

            hits = np.isin(
                ids, list(positive)
            ).astype(np.float32)

            recall.append(
                float(hits.sum()) / len(positive)
            )

            discounts = (
                1 / np.log2(np.arange(count) + 2)
            )

            dcg = float(
                np.dot(hits, discounts)
            )

            ideal = float(
                discounts[
                    :min(len(positive), count)
                ].sum()
            )

            ndcg.append(dcg / ideal)

    return {
        f"Recall@{k}": (
            float(np.mean(recall)) if recall else 0.0
        ),
        f"NDCG@{k}": (
            float(np.mean(ndcg)) if ndcg else 0.0
        ),
        "evaluated_users": len(recall)
    }


def positive_keys(train, n_items):
    keys = (
        train.user_id.to_numpy(np.int64) * n_items
        + train.note_id.to_numpy(np.int64)
    )

    return np.sort(np.unique(keys))


def sample_neg(
    users,
    keys,
    n_items,
    published,
    sample_time,
    rng
):
    """
    批量负采样：
    1. 不是训练集已知正反馈
    2. 正样本发生时已经发布
    """

    users = np.asarray(users, dtype=np.int64)
    sample_time = np.asarray(
        sample_time, dtype=np.int64
    )

    out = rng.integers(
        n_items,
        size=len(users),
        dtype=np.int64
    )

    for _ in range(10000):
        pair = users * n_items + out

        loc = np.searchsorted(keys, pair)

        present = (
            (loc < len(keys))
            & (
                keys[np.minimum(loc, len(keys) - 1)]
                == pair
            )
        )

        invalid = (
            present
            | (published[out] > sample_time)
        )

        if not invalid.any():
            return out

        out[invalid] = rng.integers(
            n_items,
            size=int(invalid.sum())
        )

    raise RuntimeError(
        "Unable to find eligible negative items"
    )