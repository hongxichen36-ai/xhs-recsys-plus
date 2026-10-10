"""
五路召回结果合并与去重

输入目录：
    data/raw_v5/

支持：
1. recall_valid.csv / recall_test.csv（长表）
2. 每个通道独立的 CSV 文件

输出：
    data/raw_v5/merged_recall_valid.csv
    data/raw_v5/merged_recall_test.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =====================================================
# 1. 配置
# =====================================================

DATA_DIR = Path("data/raw_v6")

CHANNELS = [
    "MostPopular",
    "ItemCF",
    "BPR-MF",
    "LightGCN",
    "TwoTower"
]

FILE_PREFIX = {
    "MostPopular": "mostpopular_decay",
    "ItemCF": "itemcf_weighted",
    "BPR-MF": "bprmf",
    "LightGCN": "lightgcn",
    "TwoTower": "twotower"
}

SPLITS = ["train", "valid", "test"]

# Reciprocal Rank Fusion 平滑参数
RRF_K = 60


# =====================================================
# 2. 读取 CSV / Parquet
# =====================================================

def read_file(stem):

    for suffix in [".parquet", ".csv"]:

        path = DATA_DIR / (stem + suffix)

        if path.exists():

            print(f"读取文件: {path}")

            if suffix == ".parquet":
                return pd.read_parquet(path)

            return pd.read_csv(path)

    return None


# =====================================================
# 3. 标准化召回结果
# =====================================================

def normalize_recall(df, channel=None):

    df = df.copy()

    # 兼容 score / recall_score
    if "recall_score" not in df.columns:

        if "score" in df.columns:
            df = df.rename(
                columns={"score": "recall_score"}
            )

        else:
            raise ValueError(
                "缺少 score 或 recall_score 字段"
            )

    if channel is not None:
        df["channel"] = channel

    required = [
        "user_id",
        "note_id",
        "channel",
        "recall_score"
    ]

    for col in required:

        if col not in df.columns:
            raise ValueError(
                f"缺少字段: {col}"
            )

    # 转换数据类型
    df["user_id"] = pd.to_numeric(
        df["user_id"],
        errors="raise"
    ).astype("int64")

    df["note_id"] = pd.to_numeric(
        df["note_id"],
        errors="raise"
    ).astype("int64")

    df["recall_score"] = pd.to_numeric(
        df["recall_score"],
        errors="raise"
    )

    df["channel"] = (
        df["channel"].astype(str)
    )

    # 删除完全缺失的得分记录
    df = df.dropna(
        subset=["recall_score"]
    )

    # 排除无限分数
    df = df[
        np.isfinite(df["recall_score"])
    ].copy()

    # -------------------------------------------------
    # 同一个通道内部先去重
    # -------------------------------------------------

    if "rank" in df.columns:

        df["rank"] = pd.to_numeric(
            df["rank"], errors="coerce"
        )

        df = df.sort_values(
            ["user_id", "channel", "rank",
             "recall_score"],
            ascending=[True, True, True, False]
        )

    else:

        df = df.sort_values(
            ["user_id", "channel", "recall_score"],
            ascending=[True, True, False]
        )

    df = df.drop_duplicates(
        ["user_id", "note_id", "channel"],
        keep="first"
    )

    # 根据保留后的召回得分重新计算排名
    df = df.sort_values(
        ["user_id", "channel", "recall_score",
         "note_id"],
        ascending=[True, True, False, True]
    )

    df["rank"] = (
        df.groupby(["user_id", "channel"])
        .cumcount() + 1
    )

    return df[
        [
            "user_id",
            "note_id",
            "channel",
            "recall_score",
            "rank"
        ]
    ].reset_index(drop=True)


# =====================================================
# 4. 加载某个数据切分的五路召回
# =====================================================

def load_recall(split):

    # 先检查已经合并的长表
    combined = read_file(
        f"recall_{split}"
    )

    if combined is not None:

        print(
            f"{split}: 使用已有长表召回结果"
        )

        return normalize_recall(combined)

    # 否则读取五个独立通道文件
    all_frames = []

    for channel in CHANNELS:

        prefix = FILE_PREFIX[channel]

        df = read_file(
            f"{prefix}_{split}"
        )

        if df is None:
            raise FileNotFoundError(
                f"缺少 {channel} 的 {split} 召回结果。"
            )

        frame = normalize_recall(
            df, channel=channel
        )

        all_frames.append(frame)

    combined = pd.concat(
        all_frames,
        ignore_index=True
    )

    return normalize_recall(combined)


# =====================================================
# 5. 召回结果合并与去重
# =====================================================

def merge_recall(df):

    print(
        f"\n合并前召回记录数: {len(df):,}"
    )

    # 每个用户-笔记-通道必须唯一
    assert not df.duplicated(
        ["user_id", "note_id", "channel"]
    ).any()

    # -------------------------------------------------
    # 5.1 计算通道内归一化分数
    # -------------------------------------------------

    group = df.groupby(
        ["user_id", "channel"]
    )["recall_score"]

    min_score = group.transform("min")
    max_score = group.transform("max")

    delta = max_score - min_score

    df = df.copy()

    df["norm_score"] = np.where(
        delta > 0,
        (df["recall_score"] - min_score)
        / delta.replace(0, 1),
        1.0
    )

    # -------------------------------------------------
    # 5.2 RRF 排名融合
    # -------------------------------------------------

    df["rrf_part"] = (
        1.0 / (RRF_K + df["rank"])
    )

    # -------------------------------------------------
    # 5.3 各通道结果转宽表
    # -------------------------------------------------

    base = df[
        ["user_id", "note_id"]
    ].drop_duplicates()

    base = base.set_index(
        ["user_id", "note_id"]
    )

    for channel in CHANNELS:

        part = df[
            df["channel"] == channel
        ].copy()

        part["hit"] = 1

        part = part[
            [
                "user_id",
                "note_id",
                "recall_score",
                "norm_score",
                "rank",
                "hit",
                "rrf_part"
            ]
        ]

        part = part.set_index(
            ["user_id", "note_id"]
        )

        part = part.rename(columns={
            "recall_score":
                f"{channel}_score",
            "norm_score":
                f"{channel}_norm_score",
            "rank":
                f"{channel}_rank",
            "hit":
                f"{channel}_hit",
            "rrf_part":
                f"{channel}_rrf"
        })

        base = base.join(
            part,
            how="left"
        )

    result = base.reset_index()

    # -------------------------------------------------
    # 5.4 缺失通道填充
    # -------------------------------------------------

    for channel in CHANNELS:

        # 未命中标记为 0
        result[f"{channel}_hit"] = (
            result[f"{channel}_hit"]
            .fillna(0)
            .astype("int8")
        )

        # 未命中的 rank 使用 -1
        result[f"{channel}_rank"] = (
            result[f"{channel}_rank"]
            .fillna(-1)
            .astype("int32")
        )

        # 未命中的归一化分数置 0
        result[f"{channel}_norm_score"] = (
            result[f"{channel}_norm_score"]
            .fillna(0.0)
        )

        # 原始分数保持 NaN，区分真实 0 分
        # 和没有被该通道召回

        result[f"{channel}_rrf"] = (
            result[f"{channel}_rrf"]
            .fillna(0.0)
        )

    # -------------------------------------------------
    # 5.5 融合统计特征
    # -------------------------------------------------

    hit_cols = [
        f"{c}_hit"
        for c in CHANNELS
    ]

    rrf_cols = [
        f"{c}_rrf"
        for c in CHANNELS
    ]

    norm_cols = [
        f"{c}_norm_score"
        for c in CHANNELS
    ]

    # 命中通道总数
    result["channel_count"] = (
        result[hit_cols].sum(axis=1)
        .astype("int8")
    )

    # RRF 融合得分
    result["rrf_score"] = (
        result[rrf_cols].sum(axis=1)
    )

    # 归一化召回得分最大值、平均值
    result["max_norm_score"] = (
        result[norm_cols].max(axis=1)
    )

    result["mean_norm_score"] = (
        result[norm_cols].sum(axis=1)
        / result["channel_count"].clip(lower=1)
    )

    # -------------------------------------------------
    # 5.6 按融合得分排序
    # -------------------------------------------------

    result = result.sort_values(
        [
            "user_id",
            "rrf_score",
            "channel_count",
            "note_id"
        ],
        ascending=[True, False, False, True]
    ).reset_index(drop=True)

    result["merged_rank"] = (
        result.groupby("user_id")
        .cumcount() + 1
    )

    # -------------------------------------------------
    # 5.7 质量检查
    # -------------------------------------------------

    assert not result.duplicated(
        ["user_id", "note_id"]
    ).any()

    assert result["channel_count"].between(
        1, len(CHANNELS)
    ).all()

    print(
        f"合并后唯一候选数: {len(result):,}"
    )

    print(
        "平均每个用户的候选数:",
        round(
            result.groupby("user_id").size().mean(),
            2
        )
    )

    print(
        "\n候选笔记的通道命中分布:"
    )

    print(
        result["channel_count"]
        .value_counts()
        .sort_index()
    )

    return result


# =====================================================
# 6. 执行入口
# =====================================================

def main():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for split in SPLITS:

        print(
            "\n" + "=" * 55
        )

        print(
            f"开始处理: {split}"
        )

        recall_df = load_recall(split)

        merged_df = merge_recall(
            recall_df
        )

        output_path = (
            DATA_DIR
            / f"merged_recall_{split}.csv"
        )

        merged_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"已保存: {output_path}"
        )

    print(
        "\n五路召回结果合并完成！"
    )


if __name__ == "__main__":
    main()