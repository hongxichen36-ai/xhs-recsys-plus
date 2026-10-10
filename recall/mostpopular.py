import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================
# 1. 项目配置
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw_v6"

SECONDS_PER_DAY = 86400


# =====================================================
# 2. 数据读取
# =====================================================

def read_table(root, name):

    parquet = root / f"{name}.parquet"
    csv = root / f"{name}.csv"

    if parquet.exists():
        return pd.read_parquet(parquet)

    if csv.exists():
        return pd.read_csv(csv)

    raise FileNotFoundError(
        f"找不到数据文件: {name}"
    )


# =====================================================
# 3. 时间衰减 MostPopular
# =====================================================

class TimeDecayMostPopular:

    def __init__(
        self,
        action_half_life=3.0,
        publish_half_life=7.0,
        freshness_weight=0.15,
        like_weight=0.5,
        collect_weight=1.0,
        comment_weight=1.5
    ):

        self.action_half_life = action_half_life
        self.publish_half_life = publish_half_life

        self.freshness_weight = freshness_weight

        self.like_weight = like_weight
        self.collect_weight = collect_weight
        self.comment_weight = comment_weight

        self.scores = None

    # -------------------------------------------------
    # 3.1 计算时间衰减热度
    # -------------------------------------------------

    def fit(self, history, notes, cutoff):

        n_items = len(notes)

        history = history.loc[
            history.timestamp <= cutoff
        ].copy()

        if history.empty:
            raise ValueError("历史交互为空")

        note_ids = history.note_id.to_numpy(
            dtype=np.int64
        )

        timestamps = history.timestamp.to_numpy(
            dtype=np.int64
        )

        publish_ts = notes.publish_ts.to_numpy(
            dtype=np.int64
        )

        if np.any(
            timestamps < publish_ts[note_ids]
        ):
            raise ValueError(
                "存在笔记发布前的点击"
            )

        # ---------------------------------------------
        # 计算用户行为权重
        # ---------------------------------------------

        weights = np.ones(
            len(history),
            dtype=np.float64
        )

        behavior_weights = {
            "liked": self.like_weight,
            "collected": self.collect_weight,
            "commented": self.comment_weight
        }

        for column, weight in behavior_weights.items():

            if column not in history.columns:
                raise ValueError(
                    f"缺少行为字段: {column}"
                )

            values = history[column].to_numpy(
                dtype=np.float64
            )

            weights += weight * values

        # ---------------------------------------------
        # 历史点击时间衰减
        # ---------------------------------------------

        age_days = (
            cutoff - timestamps
        ) / SECONDS_PER_DAY

        action_decay = np.exp(
            -np.log(2)
            * age_days
            / self.action_half_life
        )

        weighted_clicks = (
            weights * action_decay
        )

        # ---------------------------------------------
        # 按笔记聚合历史热度
        # ---------------------------------------------

        heat = np.bincount(
            note_ids,
            weights=weighted_clicks,
            minlength=n_items
        )

        # ---------------------------------------------
        # 笔记发布时间新鲜度
        # ---------------------------------------------

        publish_age = np.maximum(
            0,
            (cutoff - publish_ts)
            / SECONDS_PER_DAY
        )

        freshness = np.exp(
            -np.log(2)
            * publish_age
            / self.publish_half_life
        )

        # ---------------------------------------------
        # 最终得分
        # ---------------------------------------------

        scores = (
            np.log1p(heat)
            + self.freshness_weight * freshness
        )

        # 尚未发布的笔记不可推荐
        eligible = publish_ts <= cutoff

        scores[~eligible] = -np.inf

        self.scores = scores.astype(
            np.float32
        )

        print("\n========== 热度统计 ==========")
        print("历史交互数:", len(history))
        print("推荐时间:", cutoff)
        print("可推荐笔记数:", eligible.sum())
        print("非零热度笔记数:", np.count_nonzero(heat))
        print("最高时间衰减热度:", heat.max())

        return self

    # -------------------------------------------------
    # 3.2 批量用户打分
    # -------------------------------------------------

    def score(self, user_ids):

        if self.scores is None:
            raise RuntimeError(
                "模型尚未 fit"
            )

        return np.broadcast_to(
            self.scores,
            (len(user_ids), len(self.scores))
        )


# =====================================================
# 4. Top-K 召回与导出
# =====================================================

def export_recall(
    model,
    history,
    users,
    notes,
    output_path,
    topk=200
):

    seen = (
        history.groupby("user_id")
        .note_id.agg(set)
        .to_dict()
    )

    scores = model.scores
    n_items = len(notes)

    records = []

    for uid in users.user_id.to_numpy():

        user_scores = scores.copy()

        # 排除历史已经点击的笔记
        old_items = seen.get(int(uid), set())

        if old_items:
            user_scores[list(old_items)] = -np.inf

        available = np.flatnonzero(
            np.isfinite(user_scores)
        )

        k = min(topk, len(available))

        if k == 0:
            continue

        # Top-K 选择
        selected = available[
            np.argpartition(
                -user_scores[available],
                k - 1
            )[:k]
        ]

        # 按分数排序，note_id 用于稳定打破平分
        order = np.lexsort(
            (selected, -user_scores[selected])
        )

        selected = selected[order]

        records.append(
            pd.DataFrame({
                "user_id": np.full(
                    k, uid, dtype=np.int32
                ),
                "note_id": selected.astype(
                    np.int32
                ),
                "score": user_scores[selected],
                "rank": np.arange(
                    1, k + 1,
                    dtype=np.int16
                )
            })
        )

    result = pd.concat(
        records,
        ignore_index=True
    )

    assert not result.duplicated(
        ["user_id", "note_id"]
    ).any()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_parquet(
        output_path,
        index=False
    )

    print(
        f"\n已导出: {output_path}"
    )

    print(
        "召回记录数:",
        len(result)
    )

    print(
        "召回用户数:",
        result.user_id.nunique()
    )

    return result


# =====================================================
# 5. Recall / NDCG 评价
# =====================================================

def evaluate(
    recall,
    history,
    future,
    notes,
    cutoff,
    k=20
):

    # 评价时已发布的笔记
    available_notes = set(
        notes.loc[
            notes.publish_ts <= cutoff,
            "note_id"
        ]
    )

    # 历史已交互
    seen = (
        history.groupby("user_id")
        .note_id.agg(set)
        .to_dict()
    )

    # 未来正反馈
    truth = (
        future.groupby("user_id")
        .note_id.agg(set)
        .to_dict()
    )

    # 每个用户的 Top-K
    ranked = (
        recall.loc[
            recall["rank"] <= k
        ]
        .sort_values(
            ["user_id", "rank"]
        )
    )

    predictions = (
        ranked.groupby("user_id")
        .note_id.agg(list)
        .to_dict()
    )

    recalls = []
    ndcgs = []

    for uid, positives in truth.items():

        gt = (
            positives
            & available_notes
        ) - seen.get(uid, set())

        if not gt:
            continue

        recommended = predictions.get(
            uid, []
        )

        hits = np.array(
            [
                int(item in gt)
                for item in recommended
            ],
            dtype=np.float64
        )

        recalls.append(
            hits.sum() / len(gt)
        )

        discount = (
            1.0
            / np.log2(
                np.arange(len(hits)) + 2
            )
        )

        dcg = np.sum(
            hits * discount
        )

        ideal_count = min(
            k, len(gt)
        )

        idcg = np.sum(
            1.0
            / np.log2(
                np.arange(ideal_count) + 2
            )
        )

        ndcgs.append(
            dcg / idcg
        )

    return {
        "users": len(recalls),
        f"Recall@{k}": float(
            np.mean(recalls)
        ) if recalls else 0.0,
        f"NDCG@{k}": float(
            np.mean(ndcgs)
        ) if ndcgs else 0.0
    }


# =====================================================
# 6. 主程序
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR
    )

    parser.add_argument(
        "--k",
        type=int,
        default=20
    )

    parser.add_argument(
        "--topk-dump",
        type=int,
        default=200
    )

    parser.add_argument(
        "--action-half-life",
        type=float,
        default=3.0
    )

    parser.add_argument(
        "--publish-half-life",
        type=float,
        default=7.0
    )

    parser.add_argument(
        "--freshness-weight",
        type=float,
        default=0.15
    )

    parser.add_argument(
        "--like-weight",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--collect-weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--comment-weight",
        type=float,
        default=1.5
    )

    args = parser.parse_args()

    if (
        args.action_half_life <= 0
        or args.publish_half_life <= 0
        or args.topk_dump <= 0
        or args.k <= 0
    ):
        raise ValueError(
            "半衰期和 Top-K 必须大于 0"
        )

    users = read_table(
        args.data, "users"
    )

    notes = read_table(
        args.data, "notes"
    )

    train = read_table(
        args.data, "train"
    )

    valid = read_table(
        args.data, "valid"
    )

    test = read_table(
        args.data, "test"
    )

    # ---------------------------------------------
    # 验证阶段
    # ---------------------------------------------

    # 仅使用 train 计算热度
    valid_history = train

    valid_cutoff = int(
        train.timestamp.max()
    )

    # ---------------------------------------------
    # 测试阶段
    # ---------------------------------------------

    # 使用 train + valid 更新热度
    test_history = pd.concat(
        [train, valid],
        ignore_index=True
    )

    test_cutoff = int(
        valid.timestamp.max()
    )

    stages = [
        (
            "valid",
            valid_history,
            valid,
            valid_cutoff
        ),
        (
            "test",
            test_history,
            test,
            test_cutoff
        )
    ]

    for split, history, future, cutoff in stages:

        print(
            f"\n========== TimeDecay MostPopular: {split} =========="
        )

        model = TimeDecayMostPopular(
            action_half_life=args.action_half_life,
            publish_half_life=args.publish_half_life,
            freshness_weight=args.freshness_weight,
            like_weight=args.like_weight,
            collect_weight=args.collect_weight,
            comment_weight=args.comment_weight
        )

        model.fit(
            history,
            notes,
            cutoff
        )

        output_path = (
            args.data
            / f"mostpopular_decay_{split}.parquet"
        )

        # 只为目标集合中有反馈的用户导出候选，
        # 与原来五路召回实验的评价用户集合一致。
        target_users = users.loc[
            users.user_id.isin(future.user_id.unique())
        ]

        recall = export_recall(
            model,
            history,
            target_users,
            notes,
            output_path,
            topk=max(args.k, args.topk_dump)
        )

        metrics = evaluate(
            recall,
            history,
            future,
            notes,
            cutoff,
            k=args.k
        )

        print(
            f"\n{split} 评价:",
            metrics
        )

    print(
        "\nTimeDecay MostPopular 运行完成！"
    )


if __name__ == "__main__":
    main()