from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw_v6"
OUT = ROOT / "rank" / "output_v6"


def read(folder, name, columns=None):
    p = folder / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p, columns=columns)
    return pd.read_csv(
        folder / f"{name}.csv",
        usecols=columns
    )


notes = read(
    DATA, "notes",
    ["note_id", "subtopic", "publish_ts"]
)

train = read(
    DATA, "train", ["user_id", "note_id", "timestamp"]
)

valid = read(
    DATA, "valid", ["user_id", "note_id"]
)

cutoff = int(train.timestamp.max())

# 用户主兴趣只能从训练历史确定
history = (
    train[["user_id", "note_id"]]
    .drop_duplicates()
    .merge(notes[["note_id", "subtopic"]], on="note_id")
)

primary = (
    history.groupby(["user_id", "subtopic"])
    .size()
    .reset_index(name="count")
    .sort_values(
        ["user_id", "count", "subtopic"],
        ascending=[True, False, True]
    )
    .drop_duplicates("user_id")
    .set_index("user_id")["subtopic"]
)

# 构建合法验证真值
truth = (
    valid.drop_duplicates()
    .merge(notes, on="note_id", how="inner")
)

truth = truth[truth.publish_ts <= cutoff].copy()

seen = pd.MultiIndex.from_frame(
    history[["user_id", "note_id"]]
)

truth_pairs = pd.MultiIndex.from_frame(
    truth[["user_id", "note_id"]]
)

truth = truth.loc[
    ~truth_pairs.isin(seen)
].copy()

truth["primary"] = truth.user_id.map(primary)
truth = truth[truth.primary.notna()].copy()

truth["is_primary"] = (
    truth.subtopic == truth.primary
)

# 读取不同阶段候选
merged = read(
    DATA,
    "merged_recall_valid",
    ["user_id", "note_id"]
)

coarse = read(
    OUT,
    "coarse_valid_top200",
    ["user_id", "note_id"]
)

# 与粗排评价使用相同的用户群
users = set(truth.user_id) & set(coarse.user_id)
truth = truth[truth.user_id.isin(users)].copy()

for name, candidates in [
    ("五路融合全部候选", merged),
    ("LightGBM Top-200", coarse)
]:

    candidates = candidates[
        candidates.user_id.isin(users)
    ].drop_duplicates()

    candidate_keys = pd.MultiIndex.from_frame(
        candidates[["user_id", "note_id"]]
    )

    truth_keys = pd.MultiIndex.from_frame(
        truth[["user_id", "note_id"]]
    )

    temp = truth.copy()
    temp["hit"] = truth_keys.isin(candidate_keys)

    print(f"\n========== {name} ==========")

    for flag, label in [
        (True, "主兴趣"),
        (False, "非主兴趣")
    ]:
        part = temp[temp.is_primary == flag]

        hits = int(part.hit.sum())
        total = len(part)

        print(
            f"{label}: "
            f"命中 {hits:,}/{total:,}, "
            f"Micro Recall={hits/max(total, 1):.4%}"
        )

    secondary = temp[~temp.is_primary]

    if not secondary.empty:
        eligible_users = secondary.user_id.nunique()

        hit_users = (
            secondary.groupby("user_id")
            .hit.any().sum()
        )

        print(
            "有非主兴趣真实点击的用户中，"
            "至少命中一篇的比例:",
            f"{hit_users/eligible_users:.2%}"
        )