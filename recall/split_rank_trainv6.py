
"""
v6 粗排训练数据时间切分

输入：
    data/raw_v6/train.parquet
    data/raw_v6/notes.parquet

输出：
    data/raw_v6/rank_train_history.parquet
    data/raw_v6/rank_train_future.parquet

原则：
1. 全局时间切分，不随机打乱
2. History 只包含截止时刻之前的交互
3. Future 只用于粗排标签
4. 不使用 valid/test 构造粗排训练数据
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================
# 1. 路径配置
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw_v6"


def read_table(root, name):
    parquet = root / f"{name}.parquet"
    csv = root / f"{name}.csv"

    if parquet.exists():
        return pd.read_parquet(parquet)

    if csv.exists():
        return pd.read_csv(csv)

    raise FileNotFoundError(
        f"找不到 {name}.parquet 或 {name}.csv"
    )


# =====================================================
# 2. 时间切分
# =====================================================

def split_rank_train(train, ratio=0.8):

    if not 0 < ratio < 1:
        raise ValueError("ratio 必须在 0 和 1 之间")

    if train.empty:
        raise ValueError("训练集为空")

    train = train.sort_values(
        ["timestamp", "user_id", "note_id"]
    ).reset_index(drop=True)

    start_ts = int(train["timestamp"].min())
    end_ts = int(train["timestamp"].max())

    # 按时间跨度切分，不是按数据行数切分
    cutoff = int(
        start_ts + ratio * (end_ts - start_ts)
    )

    history = train.loc[
        train["timestamp"] <= cutoff
    ].copy()

    future = train.loc[
        train["timestamp"] > cutoff
    ].copy()

    if history.empty or future.empty:
        raise ValueError(
            "时间切分后历史集或未来集为空"
        )

    assert history.timestamp.max() <= cutoff
    assert future.timestamp.min() > cutoff

    assert len(history) + len(future) == len(train)

    return history, future, cutoff


# =====================================================
# 3. 数据质量检查
# =====================================================

def validate(history, future, notes, cutoff):

    publish = notes.set_index(
        "note_id"
    )["publish_ts"]

    for name, df in [
        ("History", history),
        ("Future", future)
    ]:

        if df.duplicated(
            ["user_id", "note_id"]
        ).any():
            raise ValueError(
                f"{name} 存在重复用户-笔记点击关系"
            )

        pub_ts = df.note_id.map(publish)

        if pub_ts.isna().any():
            raise ValueError(
                f"{name} 包含未知 note_id"
            )

        if (
            df.timestamp.to_numpy()
            < pub_ts.to_numpy()
        ).any():
            raise ValueError(
                f"{name} 存在发布前点击"
            )

    # 确认没有时间穿越
    assert history.timestamp.max() < future.timestamp.min()

    # 确认两组不存在重复用户-笔记关系
    h_keys = pd.MultiIndex.from_frame(
        history[["user_id", "note_id"]]
    )

    f_keys = pd.MultiIndex.from_frame(
        future[["user_id", "note_id"]]
    )

    assert not f_keys.isin(h_keys).any()

    print("\n========== 数据质量检查 ==========")
    print("时间穿越数量: 0")
    print("重复用户-笔记关系: 0")
    print("历史结束时间:", int(history.timestamp.max()))
    print("未来开始时间:", int(future.timestamp.min()))
    print("切分截止时间:", cutoff)


# =====================================================
# 4. 主程序
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR
    )

    parser.add_argument(
        "--history-ratio",
        type=float,
        default=0.8
    )

    args = parser.parse_args()

    data_dir = args.data
    data_dir.mkdir(parents=True, exist_ok=True)

    train = read_table(data_dir, "train")
    notes = read_table(data_dir, "notes")

    required = [
        "user_id",
        "note_id",
        "timestamp"
    ]

    if not set(required).issubset(train.columns):
        raise ValueError(
            f"train 缺少字段: "
            f"{set(required) - set(train.columns)}"
        )

    history, future, cutoff = split_rank_train(
        train,
        ratio=args.history_ratio
    )

    validate(
        history,
        future,
        notes,
        cutoff
    )

    # 保存 Parquet
    history_path = (
        data_dir / "rank_train_history.parquet"
    )

    future_path = (
        data_dir / "rank_train_future.parquet"
    )

    history.to_parquet(
        history_path,
        index=False
    )

    future.to_parquet(
        future_path,
        index=False
    )

    # 统计
    print("\n========== v6 粗排时间切分 ==========")

    print("原训练交互数:", f"{len(train):,}")

    print(
        "History 交互数:",
        f"{len(history):,}"
    )

    print(
        "Future 交互数:",
        f"{len(future):,}"
    )

    print(
        "History 占比:",
        f"{len(history) / len(train):.2%}"
    )

    print(
        "Future 占比:",
        f"{len(future) / len(train):.2%}"
    )

    print(
        "History 用户数:",
        history.user_id.nunique()
    )

    print(
        "Future 用户数:",
        future.user_id.nunique()
    )

    print(
        "History 笔记数:",
        history.note_id.nunique()
    )

    print(
        "Future 笔记数:",
        future.note_id.nunique()
    )

    # 推荐时刻已发布的笔记
    eligible_notes = (
        notes.publish_ts <= cutoff
    ).sum()

    print(
        "截止时刻已发布笔记:",
        eligible_notes
    )

    print("\n文件保存成功：")
    print(history_path)
    print(future_path)


if __name__ == "__main__":
    main()
