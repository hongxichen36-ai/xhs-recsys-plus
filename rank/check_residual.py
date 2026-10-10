from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "rank" / "output_v6"

for split in ("valid", "test"):

    df = pd.read_parquet(
        OUT / f"fine_{split}_top200.parquet"
    )

    diff = np.abs(
        df["fine_score"].to_numpy()
        - df["prediction"].to_numpy()
    )

    lgbm = (
        df.sort_values(
            ["user_id", "prediction", "note_id"],
            ascending=[True, False, True]
        )
        .groupby("user_id")
        .head(20)
    )

    deepfm = (
        df.sort_values(
            ["user_id", "fine_score", "note_id"],
            ascending=[True, False, True]
        )
        .groupby("user_id")
        .head(20)
    )

    a = (
        lgbm.groupby("user_id")
        .note_id.agg(set)
    )

    b = (
        deepfm.groupby("user_id")
        .note_id.agg(set)
    )

    common = a.index.intersection(b.index)

    overlap = np.mean([
        len(a[u] & b[u]) / 20
        for u in common
    ])

    print(f"\n========== {split} ==========")
    print("Mean score difference:", diff.mean())
    print("Max score difference:", diff.max())
    print("Top20 overlap:", overlap)
    print("Changed users:", sum(a[u] != b[u] for u in common))


# 查看学习后的残差门控幅度
ckpt = torch.load(
    OUT / "deepfm_residual_v6.pt",
    map_location="cpu",
    weights_only=False  # 仅加载自己生成且可信的 checkpoint
)

rho = ckpt["state_dict"]["rho"].item()
max_residual = 0.5 * abs(np.tanh(rho))

print("\nLearned rho:", rho)
print("Maximum absolute logit residual:", max_residual)