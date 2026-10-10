
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).parent))
from dump_utils import dump_recall


# =====================================================
# 1. 数据读取
# =====================================================

def read_table(root, name):
    p = root / f"{name}.parquet"

    if p.exists():
        return pd.read_parquet(p)

    return pd.read_csv(root / f"{name}.csv")


def load_data(root):
    users = read_table(root, "users")
    notes = read_table(root, "notes")
    train = read_table(root, "train")
    valid = read_table(root, "valid")
    test = read_table(root, "test")

    assert np.array_equal(
        users.user_id.to_numpy(),
        np.arange(len(users))
    )

    assert np.array_equal(
        notes.note_id.to_numpy(),
        np.arange(len(notes))
    )

    return users, notes, train, valid, test


# =====================================================
# 2. Top-K 评价
# 与原代码保持一致
# =====================================================

def evaluate(
    score_fn,
    train,
    target,
    notes,
    n_users,
    k=20,
    batch_size=64
):
    train_sets = (
        train.groupby("user_id")
        .note_id.agg(set).to_dict()
    )

    target_sets = (
        target.groupby("user_id")
        .note_id.agg(set).to_dict()
    )

    publish = notes.publish_ts.to_numpy()

    cutoff = (
        int(target.timestamp.min())
        if len(target) else 0
    )

    eligible = publish <= cutoff

    users = np.array(
        sorted(target_sets),
        dtype=np.int64
    )

    recalls = []
    ndcgs = []

    for start in range(
        0, len(users), batch_size
    ):

        uu = users[start:start + batch_size]

        scores = score_fn(uu).astype(
            np.float32, copy=True
        )

        scores[:, ~eligible] = -np.inf

        for r, u in enumerate(uu):
            seen = train_sets.get(
                int(u), set()
            )

            if seen:
                scores[r, list(seen)] = -np.inf

            gt = target_sets[int(u)].difference(
                seen
            )

            gt = {
                i for i in gt
                if eligible[i]
            }

            if not gt:
                continue

            kk = min(
                k,
                int(np.isfinite(scores[r]).sum())
            )

            if kk == 0:
                continue

            ids = np.argpartition(
                -scores[r], kk - 1
            )[:kk]

            ids = ids[
                np.argsort(-scores[r, ids])
            ]

            hits = np.fromiter(
                (i in gt for i in ids),
                dtype=np.float32
            )

            recall = (
                float(hits.sum()) / len(gt)
            )

            dcg = float(
                (
                    hits /
                    np.log2(np.arange(kk) + 2)
                ).sum()
            )

            ideal = float(
                (
                    1 /
                    np.log2(
                        np.arange(
                            min(len(gt), kk)
                        ) + 2
                    )
                ).sum()
            )

            recalls.append(recall)
            ndcgs.append(dcg / ideal)

    return {
        "users": len(recalls),
        f"Recall@{k}": (
            float(np.mean(recalls))
            if recalls else 0.0
        ),
        f"NDCG@{k}": (
            float(np.mean(ndcgs))
            if ndcgs else 0.0
        )
    }


# =====================================================
# 3. 构建用户对笔记的偏好权重
# =====================================================

def build_preference_weights(
    train,
    like_weight=1.0,
    collect_weight=1.0,
    comment_weight=1.0
):
    """
    用户偏好：
        仅点击 = 1
        点赞额外 +like_weight
        收藏额外 +collect_weight
        评论额外 +comment_weight

    仅使用 train，不使用 valid/test 行为。
    """

    df = train.copy()

    behavior_cols = [
        "liked",
        "collected",
        "commented"
    ]

    missing = [
        col for col in behavior_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "训练数据缺少行为字段: "
            + ", ".join(missing)
        )

    for col in behavior_cols:
        df[col] = pd.to_numeric(
            df[col], errors="raise"
        )

        if not df[col].isin([0, 1]).all():
            raise ValueError(
                f"{col} 必须是 0/1 字段"
            )

    # 先转换成 float32，避免整数溢出
    df["preference"] = (
        1.0
        + like_weight
        * df["liked"].astype(np.float32)
        + collect_weight
        * df["collected"].astype(np.float32)
        + comment_weight
        * df["commented"].astype(np.float32)
    ).astype(np.float32)

    if (df["preference"] <= 0).any():
        raise ValueError(
            "偏好权重必须为正"
        )

    if df.duplicated(
        ["user_id", "note_id"]
    ).any():
        raise ValueError(
            "train 中存在重复用户-笔记对，"
            "请先明确重复行为的聚合规则"
        )

    return df


# =====================================================
# 4. Weighted ItemCF 模型
# =====================================================

class WeightedItemCF:

    def __init__(
        self,
        n_users,
        n_items,
        neighbors=100,
        max_history=100,
        popularity_weight=1e-5
    ):
        self.n_users = n_users
        self.n_items = n_items

        self.neighbors = neighbors
        self.max_history = max_history

        self.popularity_weight = (
            popularity_weight
        )

        self.sim = None
        self.full_X = None
        self.popularity = None

    # -------------------------------------------------
    # 4.1 模型训练
    # -------------------------------------------------

    def fit(self, train):

        # ---------------------------------------------
        # 构建加权用户-物品矩阵
        # ---------------------------------------------

        # 仅对相似度计算限制用户历史长度
        limited = (
            train.sort_values("timestamp")
            .groupby("user_id")
            .tail(self.max_history)
        )

        X = csr_matrix(
            (
                limited.preference.to_numpy(
                    np.float32
                ),
                (
                    limited.user_id.to_numpy(),
                    limited.note_id.to_numpy()
                )
            ),
            shape=(self.n_users, self.n_items)
        )

        print(
            "加权用户-笔记矩阵:",
            X.shape
        )

        print(
            "非零交互:",
            X.nnz
        )

        # ---------------------------------------------
        # 计算加权余弦相似度
        # ---------------------------------------------

        # 每篇笔记的平方范数
        square_norm = np.asarray(
            X.power(2).sum(axis=0)
        ).ravel()

        norm = np.sqrt(
            np.maximum(square_norm, 1e-12)
        )

        # 加权共现矩阵
        sim = (X.T @ X).tocsr()

        # 不允许自己推荐自己
        sim.setdiag(0)
        sim.eliminate_zeros()

        # 加权余弦归一化
        inv_norm = 1.0 / norm

        sim = (
            sim.multiply(inv_norm[:, None])
            .multiply(inv_norm[None, :])
            .tocsr()
        )

        print(
            "加权物品共现关系数:",
            sim.nnz
        )

        # ---------------------------------------------
        # 每篇笔记保留 Top-K 邻居
        # ---------------------------------------------

        indptr = [0]
        indices = []
        data = []

        for i in range(self.n_items):

            start = sim.indptr[i]
            end = sim.indptr[i + 1]

            cols = sim.indices[start:end]
            vals = sim.data[start:end]

            if len(vals) > self.neighbors:

                keep = np.argpartition(
                    vals, -self.neighbors
                )[-self.neighbors:]

                cols = cols[keep]
                vals = vals[keep]

            # 统一采用确定性排序
            if len(vals) > 0:
                order = np.lexsort(
                    (cols, -vals)
                )

                cols = cols[order]
                vals = vals[order]

            indices.extend(cols.tolist())
            data.extend(vals.tolist())

            indptr.append(len(indices))

        self.sim = csr_matrix(
            (
                np.asarray(
                    data, dtype=np.float32
                ),
                np.asarray(
                    indices, dtype=np.int32
                ),
                np.asarray(
                    indptr, dtype=np.int64
                )
            ),
            shape=(self.n_items, self.n_items)
        )

        # ---------------------------------------------
        # 构建完整加权历史矩阵
        # ---------------------------------------------

        self.full_X = csr_matrix(
            (
                train.preference.to_numpy(
                    np.float32
                ),
                (
                    train.user_id.to_numpy(),
                    train.note_id.to_numpy()
                )
            ),
            shape=(self.n_users, self.n_items)
        )

        # ---------------------------------------------
        # 历史热度作为冷启动兜底
        # ---------------------------------------------

        # 热度使用点击人数，而不是偏好权重和
        popularity = np.bincount(
            train.note_id.to_numpy(
                np.int64
            ),
            minlength=self.n_items
        ).astype(np.float32)

        self.popularity = (
            popularity /
            max(float(popularity.max()), 1.0)
        )

        print(
            "保留物品相似关系:",
            self.sim.nnz
        )

        return self

    # -------------------------------------------------
    # 4.2 推荐打分
    # -------------------------------------------------

    def score(self, user_ids):

        if self.sim is None:
            raise RuntimeError(
                "模型尚未 fit"
            )

        scores = (
            self.full_X[user_ids] @ self.sim
        ).toarray()

        scores = scores.astype(
            np.float32,
            copy=False
        )

        # 冷用户、稀疏用户的热度兜底
        scores += (
            self.popularity_weight
            * self.popularity[None, :]
        )

        return scores


# =====================================================
# 5. 主程序
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("./data/raw_v6")
    )

    parser.add_argument(
        "--k",
        type=int,
        default=20
    )

    parser.add_argument(
        "--neighbors",
        type=int,
        default=200
    )

    parser.add_argument(
        "--max-history",
        type=int,
        default=100
    )

    parser.add_argument(
        "--topk-dump",
        type=int,
        default=200
    )

    # ---------------------------------------------
    # 新增：行为偏好权重
    # ---------------------------------------------

    parser.add_argument(
        "--like-weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--collect-weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--comment-weight",
        type=float,
        default=1.0
    )

    args = parser.parse_args()

    users, notes, train, valid, test = load_data(
        args.data
    )

    n_users = len(users)
    n_items = len(notes)

    # ---------------------------------------------
    # 5.1 构造训练偏好
    # ---------------------------------------------

    weighted_train = build_preference_weights(
        train,
        like_weight=args.like_weight,
        collect_weight=args.collect_weight,
        comment_weight=args.comment_weight
    )

    print(
        "\n========== 偏好权重统计 =========="
    )

    print(
        weighted_train.preference.describe()
    )

    # ---------------------------------------------
    # 5.2 训练 Weighted ItemCF
    # ---------------------------------------------

    model = WeightedItemCF(
        n_users=n_users,
        n_items=n_items,
        neighbors=args.neighbors,
        max_history=args.max_history
    )

    model.fit(weighted_train)

    # 保持原来的 batch 得分接口
    def score_fn(uid):
        return model.score(uid)

    # ---------------------------------------------
    # 5.3 验证集评价
    # ---------------------------------------------

    print(
        "\n========== Weighted ItemCF =========="
    )

    valid_metrics = evaluate(
        score_fn,
        train,
        valid,
        notes,
        n_users,
        args.k
    )

    print(
        "验证集:",
        valid_metrics
    )

    # ---------------------------------------------
    # 5.4 测试集评价
    # ---------------------------------------------

    test_metrics = evaluate(
        score_fn,
        train,
        test,
        notes,
        n_users,
        args.k
    )

    print(
        "测试集:",
        test_metrics
    )

    # ---------------------------------------------
    # 5.5 导出候选结果
    # ---------------------------------------------

    # 使用新的前缀，防止覆盖旧 ItemCF 结果
    for split_name, target in (
        ("valid", valid),
        ("test", test)
    ):

        dump_recall(
            score_fn,
            split_name,
            train,
            target,
            notes,
            prefix="itemcf_weighted",
            topk=args.topk_dump
        )

    print(
        "\nWeighted ItemCF 训练与导出完成"
    )


if __name__ == "__main__":
    main()
