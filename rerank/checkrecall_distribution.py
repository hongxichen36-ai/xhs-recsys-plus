
from pathlib import Path
import pandas as pd
import numpy as np

# =====================================================
# 1. 路径配置
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

data_dir = ROOT / "data" / "raw_v6"
rank_dir = ROOT / "rank" / "output_v6"

SPLIT = "test"


# =====================================================
# 2. 自动读取 CSV / Parquet
# =====================================================

def read_data(folder, name):

    parquet = folder / f"{name}.parquet"
    csv = folder / f"{name}.csv"

    if parquet.exists():
        return pd.read_parquet(parquet)

    if csv.exists():
        return pd.read_csv(csv)

    raise FileNotFoundError(
        f"找不到文件: {name}"
    )


# =====================================================
# 3. 读取笔记主题信息
# =====================================================

notes = read_data(
    data_dir, "notes"
)

topic_map = notes.set_index(
    "note_id"
)["subtopic"]


# =====================================================
# 4. 读取不同阶段的候选
# =====================================================

# 五路融合后的全部候选
merged = read_data(
    data_dir,
    f"merged_recall_{SPLIT}"
)

# LightGBM 粗排 Top-200
coarse = read_data(
    rank_dir,
    f"coarse_{SPLIT}_top200"
)

# DeepFM 精排 Top-200
fine = read_data(
    rank_dir,
    f"fine_{SPLIT}_top200_v2"
)

# 使用相同的用户集合进行比较
user_ids = set(
    fine["user_id"].unique()
)


# =====================================================
# 5. 选择 Top-K 并关联细分主题
# =====================================================

def prepare(frame, score_col=None, k=None):

    df = frame[
        ["user_id", "note_id"]
        + ([score_col] if score_col else [])
    ].copy()

    # 确保各阶段比较同一批用户
    df = df[
        df["user_id"].isin(user_ids)
    ].copy()

    if df.duplicated(
        ["user_id", "note_id"]
    ).any():
        raise ValueError(
            "存在重复用户-笔记组合"
        )

    # 根据模型得分进行排序
    if score_col is not None:

        df = df.sort_values(
            ["user_id", score_col, "note_id"],
            ascending=[True, False, True]
        )

    # 选取 Top-K
    if k is not None:

        df = (
            df.groupby("user_id", sort=False)
            .head(k)
            .copy()
        )

    # 关联笔记细分主题
    df["subtopic"] = (
        df["note_id"].map(topic_map)
    )

    if df["subtopic"].isna().any():
        raise ValueError(
            "部分笔记找不到细分主题"
        )

    return df


# =====================================================
# 6. 主题分布统计
# =====================================================

def report(name, df):

    # 每个用户实际候选数量
    user_counts = (
        df.groupby("user_id")
        .size()
    )

    # 每个用户的不同细分主题数
    topic_counts = (
        df.groupby("user_id")
        .subtopic.nunique()
    )

    # 每个用户-细分主题下的笔记数量
    topic_distribution = (
        df.groupby(
            ["user_id", "subtopic"]
        )
        .size()
    )

    # 每个用户最多的细分主题笔记数量
    dominant_counts = (
        topic_distribution
        .groupby(level=0)
        .max()
    )

    # 主导主题占比
    dominant_ratio = (
        dominant_counts / user_counts
    )

    # 非主导主题笔记数量
    other_topic_count = (
        user_counts - dominant_counts
    )

    print(f"\n========== {name} ==========")

    print(
        "用户数量:",
        len(user_counts)
    )

    print(
        "平均每用户候选数:",
        round(user_counts.mean(), 4)
    )

    print(
        "平均细分主题数:",
        round(topic_counts.mean(), 4)
    )

    print(
        "仅一个细分主题的用户比例:",
        round((topic_counts == 1).mean(), 4)
    )

    print(
        "至少两个细分主题的用户比例:",
        round((topic_counts >= 2).mean(), 4)
    )

    print(
        "至少三个细分主题的用户比例:",
        round((topic_counts >= 3).mean(), 4)
    )

    print(
        "主导主题平均占比:",
        round(dominant_ratio.mean(), 4)
    )

    print(
        "平均非主导主题笔记数:",
        round(other_topic_count.mean(), 4)
    )

    print("\n细分主题数量分布:")

    distribution = pd.Series({
        "1个主题": (topic_counts == 1).sum(),
        "2个主题": (topic_counts == 2).sum(),
        "3个主题": (topic_counts == 3).sum(),
        "4个及以上": (topic_counts >= 4).sum()
    })

    print(distribution)

    return {
        "stage": name,
        "users": len(user_counts),
        "avg_candidates": user_counts.mean(),
        "avg_subtopics": topic_counts.mean(),
        "single_topic_ratio": (
            topic_counts == 1
        ).mean(),
        "dominant_topic_ratio": (
            dominant_ratio.mean()
        ),
        "avg_other_topic_items": (
            other_topic_count.mean()
        )
    }


# =====================================================
# 7. 四个阶段逐一比较
# =====================================================

results = []

# 第一阶段：五路融合全部候选
merged_all = prepare(
    merged
)

results.append(
    report(
        "五路融合全部候选",
        merged_all
    )
)

# 第二阶段：LightGBM Top-200
coarse_top200 = prepare(
    coarse,
    score_col="prediction",
    k=200
)

results.append(
    report(
        "LightGBM Top-200",
        coarse_top200
    )
)

# 第三阶段：LightGBM Top-20
coarse_top20 = prepare(
    coarse,
    score_col="prediction",
    k=20
)

results.append(
    report(
        "LightGBM Top-20",
        coarse_top20
    )
)

# 第四阶段：DeepFM Top-20
fine_top20 = prepare(
    fine,
    score_col="fine_score",
    k=20
)

results.append(
    report(
        "DeepFM Top-20",
        fine_top20
    )
)


# =====================================================
# 8. 汇总对比
# =====================================================

summary = pd.DataFrame(results)

print("\n" + "=" * 70)
print("四阶段细分主题分布对比")
print("=" * 70)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# 保存统计结果
output = (
    rank_dir / f"topic_distribution_{SPLIT}.csv"
)

summary.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("\n统计结果已保存:", output)
