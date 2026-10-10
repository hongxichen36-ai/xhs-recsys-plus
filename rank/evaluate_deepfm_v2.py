from pathlib import Path

import pandas as pd

from deepfm_residual_v2 import (
    read,
    truth_at_snapshot,
    evaluate
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw_v6"
OUT_DIR = ROOT / "rank" / "output_v6"


def main():

    notes = read(DATA_DIR, "notes")
    train = read(DATA_DIR, "train")
    valid = read(DATA_DIR, "valid")
    test = read(DATA_DIR, "test")

    # 与当前测试阶段使用的历史快照一致
    history = pd.concat(
        [train, valid],
        ignore_index=True
    )

    cutoff = int(history.timestamp.max())

    truth = truth_at_snapshot(
        test, history, notes, cutoff
    )

    fine = read(
        OUT_DIR,
        "fine_test_top200_v2"
    )

    print("========== v6 Final Test ==========")
    print("评价用户数:", len(truth))
    print("精排候选数:", len(fine))

    print(
        f"\n{'K':<8}"
        f"{'Model':<16}"
        f"{'Recall':<14}"
        f"{'NDCG':<14}"
    )

    for k in [20, 100, 200]:

        for model_name, score_col in [
            ("LightGBM", "prediction"),
            ("DeepFM_v2", "fine_score")
        ]:

            recall, ndcg = evaluate(
                fine,
                truth,
                score_col,
                k
            )

            print(
                f"{k:<8}"
                f"{model_name:<16}"
                f"{recall:<14.6f}"
                f"{ndcg:<14.6f}"
            )


if __name__ == "__main__":
    main()