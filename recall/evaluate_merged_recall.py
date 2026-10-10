
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw_v6"

CHANNELS = {
    "MostPopular": "mostpopular_decay",
    "ItemCF": "itemcf_weighted",
    "BPR-MF": "bprmf",
    "LightGCN": "lightgcn",
    "TwoTower": "twotower",
}

TOPKS = [20, 100, 200, 500]
RRF_K = 60


def read(name):
    for ext in [".parquet", ".csv"]:
        path = DATA_DIR / f"{name}{ext}"
        if path.exists():
            return (pd.read_parquet(path)
                    if ext == ".parquet"
                    else pd.read_csv(path))
    raise FileNotFoundError(
        f"找不到 {name}.parquet 或 .csv"
    )


def get_history(split):
    train = read("train")
    if split == "valid":
        return train, read("valid")

    valid = read("valid")
    history = pd.concat(
        [train, valid], ignore_index=True
    )
    return history, read("test")


def load_channel(channel, prefix, split,
                 eligible, seen_keys, n_items):
    df = read(f"{prefix}_{split}")

    if "recall_score" in df:
        df = df.rename(
            columns={"recall_score": "score"}
        )

    required = {"user_id", "note_id", "score"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{prefix}_{split} 缺少 {required - set(df.columns)}"
        )

    df = df[["user_id", "note_id", "score"]].copy()
    df["user_id"] = df.user_id.astype(np.int64)
    df["note_id"] = df.note_id.astype(np.int64)

    if not df.note_id.between(0, n_items - 1).all():
        raise ValueError(f"{channel} 存在无效 note_id")

    if not np.isfinite(df.score).all():
        raise ValueError(f"{channel} 存在无效分数")

    # 通道内重复关系保留最高分
    df = df.sort_values(
        ["user_id", "score", "note_id"],
        ascending=[True, False, True]
    ).drop_duplicates(["user_id", "note_id"])

    # 排除未发布的笔记
    df = df.loc[
        eligible[df.note_id.to_numpy()]
    ].copy()

    # 排除历史正反馈
    keys = (
        df.user_id.to_numpy() * n_items
        + df.note_id.to_numpy()
    )
    df = df.loc[~np.isin(keys, seen_keys)].copy()

    # 过滤后重新排名
    df = df.sort_values(
        ["user_id", "score", "note_id"],
        ascending=[True, False, True]
    )
    df["rank"] = (
        df.groupby("user_id").cumcount() + 1
    ).astype(np.int32)

    df["channel"] = channel
    df["rrf_part"] = 1.0 / (RRF_K + df["rank"])

    return df.reset_index(drop=True)


def evaluate(pred, truth, users, k=None):
    if k is not None:
        pred = pred.loc[pred["rank"] <= k]

    matches = pred[["user_id", "note_id"]].merge(
        truth[["user_id", "note_id"]],
        on=["user_id", "note_id"],
        how="inner"
    )

    positives = truth.groupby("user_id").size()
    hits = matches.groupby("user_id").size()

    recalls = (
        hits.reindex(users, fill_value=0)
        / positives.reindex(users)
    )

    hitrate = (
        hits.reindex(users, fill_value=0) > 0
    ).mean()

    return {
        "Recall": float(recalls.mean()),
        "HitRate": float(hitrate),
        "PositiveHits": len(matches),
        "CandidateCount": len(pred)
    }


def run(split):
    print(f"\n========== {split.upper()} ==========")

    notes = read("notes")
    history, future = get_history(split)

    n_items = len(notes)
    cutoff = int(history.timestamp.max())
    eligible = notes.publish_ts.to_numpy() <= cutoff

    history = history[
        ["user_id", "note_id"]
    ].drop_duplicates()

    seen_keys = np.unique(
        history.user_id.to_numpy(np.int64) * n_items
        + history.note_id.to_numpy(np.int64)
    )

    truth = future[
        ["user_id", "note_id"]
    ].drop_duplicates().copy()

    truth = truth.loc[
        eligible[truth.note_id.to_numpy()]
    ]

    truth_keys = (
        truth.user_id.to_numpy(np.int64) * n_items
        + truth.note_id.to_numpy(np.int64)
    )
    truth = truth.loc[
        ~np.isin(truth_keys, seen_keys)
    ].copy()

    users = np.sort(truth.user_id.unique())

    if len(users) == 0:
        raise ValueError("没有符合条件的评价用户")

    print("历史截止时间:", cutoff)
    print("评价用户数:", len(users))
    print("未来正反馈:", len(truth))

    frames = []
    report = []

    for channel, prefix in CHANNELS.items():
        df = load_channel(
            channel, prefix, split,
            eligible, seen_keys, n_items
        )

        # 与其他通道使用同一评价用户集合
        df = df.loc[df.user_id.isin(users)]

        # 限制每路最多 Top-200
        df = df.loc[df["rank"] <= 200].copy()

        m = evaluate(df, truth, users)
        report.append({"Model": channel, **m})

        frames.append(df)
        print(f"{channel}: {len(df):,} candidates")

    # 合并去重，保留来源信息
    all_recalls = pd.concat(
        frames, ignore_index=True
    )

    merged = all_recalls.groupby(
        ["user_id", "note_id"],
        as_index=False
    ).agg(
        rrf_score=("rrf_part", "sum"),
        channel_count=("channel", "nunique")
    )

    merged = merged.sort_values(
        ["user_id", "rrf_score", "note_id"],
        ascending=[True, False, True]
    )

    merged["rank"] = (
        merged.groupby("user_id").cumcount() + 1
    )

    m = evaluate(merged, truth, users)
    report.append({"Model": "Merged_All", **m})

    for k in TOPKS:
        m = evaluate(merged, truth, users, k=k)
        report.append({"Model": f"RRF@{k}", **m})

    # 计算每路独有的未来正反馈贡献
    positive_channels = all_recalls[
        ["user_id", "note_id", "channel"]
    ].merge(
        truth,
        on=["user_id", "note_id"],
        how="inner"
    )

    positive_counts = positive_channels.groupby(
        ["user_id", "note_id"]
    ).channel.nunique()

    exclusive_keys = positive_counts[
        positive_counts == 1
    ].index

    exclusive = positive_channels.set_index(
        ["user_id", "note_id"]
    )

    exclusive = exclusive.loc[
        exclusive.index.isin(exclusive_keys)
    ]

    exclusive_report = (
        exclusive.channel.value_counts()
        .reindex(CHANNELS.keys(), fill_value=0)
    )

    result = pd.DataFrame(report)

    print("\n========== 召回质量 ==========")
    print(result.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    ))

    print("\n========== 独有正反馈贡献 ==========")
    print(exclusive_report.to_string())

    print("\n平均每用户候选数:",
          round(len(merged) / len(users), 2))

    print("\n通道命中分布:")
    print(
        merged.channel_count.value_counts()
        .sort_index().to_string()
    )

    result.to_csv(
        DATA_DIR / f"recall_quality_{split}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    merged.to_parquet(
        DATA_DIR / f"merged_recall_{split}.parquet",
        index=False
    )

    print("\n报告和合并候选已保存至:", DATA_DIR)


if __name__ == "__main__":
    run("valid")
    run("test")
