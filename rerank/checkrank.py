from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw_v6"
OUT = ROOT / "rank" / "output_v6"

SPLIT = "valid"  # 优先在验证集诊断

notes = pd.read_parquet(
    DATA / "notes.parquet"
)[["note_id", "subtopic"]]

coarse = pd.read_parquet(
    OUT / f"coarse_{SPLIT}_top200.parquet"
)

fine = pd.read_parquet(
    OUT / f"fine_{SPLIT}_top200_v2.parquet"
)


def report(name, frame, score_col, k):

    df = frame.drop(
        columns=["subtopic"],
        errors="ignore"
    ).merge(
        notes,
        on="note_id",
        how="left",
        validate="many_to_one"
    )

    df = (
        df.sort_values(
            ["user_id", score_col, "note_id"],
            ascending=[True, False, True]
        )
        .groupby("user_id")
        .head(k)
    )

    topic_counts = (
        df.groupby("user_id")
        .subtopic.nunique()
    )

    # 每个用户最频繁主题占推荐列表的比例
    dominant_count = (
        df.groupby(["user_id", "subtopic"])
        .size()
        .groupby(level=0)
        .max()
    )

    list_length = df.groupby("user_id").size()

    dominant_ratio = dominant_count / list_length

    print(f"\n========== {name} ==========")

    print(
        "平均主题数量:",
        round(topic_counts.mean(), 4)
    )

    print(
        "仅一个主题用户占比:",
        round((topic_counts == 1).mean(), 4)
    )

    print(
        "主导主题平均占比:",
        round(dominant_ratio.mean(), 4)
    )


report(
    "LightGBM Top-200",
    coarse,
    "prediction",
    200
)

report(
    "LightGBM Top-20",
    coarse,
    "prediction",
    20
)

report(
    "DeepFM Top-20",
    fine,
    "fine_score",
    20
)