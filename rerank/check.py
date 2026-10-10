from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "raw_v6"
OUT = ROOT / "rank" / "output_v6"

# 只在验证集上诊断和决定修改方案
SPLIT = "valid"

notes = pd.read_parquet(
    DATA / "notes.parquet"
)[["note_id", "subtopic", "publish_ts"]]

train = pd.read_parquet(DATA / "train.parquet")
valid = pd.read_parquet(DATA / "valid.parquet")

coarse = pd.read_parquet(
    OUT / "coarse_valid_top200.parquet"
)

fine = pd.read_parquet(
    OUT / "fine_valid_top200_v2.parquet"
)

# ==========================================
# 1. 从训练历史推断用户主要兴趣
# ==========================================

history = (
    train[["user_id", "note_id"]]
    .drop_duplicates()
    .merge(
        notes[["note_id", "subtopic"]],
        on="note_id",
        how="left"
    )
)

topic_count = (
    history.groupby(["user_id", "subtopic"])
    .size()
    .rename("count")
    .reset_index()
)

# 选择历史交互最多的主题
primary = (
    topic_count.sort_values(
        ["user_id", "count", "subtopic"],
        ascending=[True, False, True]
    )
    .drop_duplicates("user_id")
    .set_index("user_id")["subtopic"]
)

# ==========================================
# 2. 准备未来验证集真实正反馈
# ==========================================

cutoff = int(train.timestamp.max())

future = (
    valid[["user_id", "note_id"]]
    .drop_duplicates()
    .merge(
        notes,
        on="note_id",
        how="left"
    )
)

# 评价时必须已经发布
future = future.loc[
    future.publish_ts <= cutoff
].copy()

# 排除历史已点击关系
seen = pd.MultiIndex.from_frame(
    history[["user_id", "note_id"]]
)

pairs = pd.MultiIndex.from_frame(
    future[["user_id", "note_id"]]
)

future = future.loc[
    ~pairs.isin(seen)
].copy()

# 仅比较有历史且有未来正反馈的用户
users = set(primary.index) & set(future.user_id)

# ==========================================
# 3. 对比各阶段的主要兴趣占比
# ==========================================

def report(name, frame, score_col=None, k=None):

    df = frame[
        ["user_id", "note_id"]
    ].copy() if score_col is None else frame[
        ["user_id", "note_id", score_col]
    ].copy()

    df = df[df.user_id.isin(users)].copy()

    if score_col is not None:
        df = df.sort_values(
            ["user_id", score_col, "note_id"],
            ascending=[True, False, True]
        )

    if k is not None:
        df = df.groupby(
            "user_id", sort=False
        ).head(k)

    df = df.merge(
        notes[["note_id", "subtopic"]],
        on="note_id",
        how="left"
    )

    df["main_topic"] = df.user_id.map(primary)

    df["is_main"] = (
        df.subtopic == df.main_topic
    ).astype(float)

    shares = df.groupby("user_id").is_main.mean()

    print(f"\n========== {name} ==========")
    print("用户数:", len(shares))
    print("平均主兴趣占比:", round(shares.mean(), 4))
    print("主兴趣占比中位数:", round(shares.median(), 4))
    print(
        "主兴趣占比 >=90% 用户比例:",
        round((shares >= 0.9).mean(), 4)
    )


report("训练历史", history)
report("未来真实点击", future)

report(
    "LightGBM Top-200",
    coarse
)

report(
    "LightGBM Top-20",
    coarse,
    score_col="prediction",
    k=20
)

report(
    "DeepFM Top-20",
    fine,
    score_col="fine_score",
    k=20
)