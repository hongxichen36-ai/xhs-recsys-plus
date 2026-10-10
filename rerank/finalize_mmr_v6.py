"""One-command V6 MMR validation selection and FINAL TEST export.

Place in: xhs-recsys/rerank/finalize_mmr_v6.py
Needs the existing V6 `rerank/rerank_mmr.py` (calc_mmr, diversity_stats,
accuracy_stats, read_table). Does not retrain anything.

Run from anywhere:
    python rerank/finalize_mmr_v6.py

Uses VALID to choose beta without looking at TEST labels. Then it freezes beta,
exports `rank/output_v6/rerank_test_top20_v2.parquet` and prints FINAL TEST metrics.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Same MMR scoring/similarity and evaluation functions as current project.
from rerank_mmr import calc_mmr, diversity_stats, accuracy_stats, read_table

ROOT = Path(__file__).resolve().parent.parent


def read_dataset(data_dir, name):
    for suffix in (".parquet", ".csv"):
        path = data_dir / f"{name}{suffix}"
        if path.exists():
            return read_table(data_dir, name)
    raise FileNotFoundError(f"Missing {name} parquet/csv in {data_dir}")


def read_predictions(output_dir, split):
    for suffix in (".parquet", ".csv"):
        path = output_dir / f"fine_{split}_top200_v2{suffix}"
        if path.exists():
            return read_table(output_dir, f"fine_{split}_top200_v2")
    raise FileNotFoundError(
        f"Missing {output_dir}/fine_{split}_top200_v2.parquet (or .csv). "
        "First export the DeepFM v2 fine-rank results on your own machine."
    )


def load_stage(data_dir, rank_dir, notes, train, valid, split):
    history = train if split == "valid" else pd.concat([train, valid], ignore_index=True)
    cutoff = int(history.timestamp.max())
    raw = read_predictions(rank_dir, split)
    required = {"user_id", "note_id", "fine_score"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Fine-rank {split} is missing {required-set(raw.columns)}")
    if raw.duplicated(["user_id", "note_id"]).any():
        raise ValueError(f"Fine-rank {split} has duplicate user-note pairs")
    if not np.isfinite(raw["fine_score"]).all():
        raise ValueError(f"Fine-rank {split} has non-finite scores")

    metadata = notes[["note_id", "category", "subtopic", "author_id", "publish_ts"]]
    frame = raw[["user_id", "note_id", "fine_score"]].merge(
        metadata, on="note_id", how="left", validate="many_to_one"
    )
    if frame[["category", "subtopic", "author_id", "publish_ts"]].isna().any().any():
        raise ValueError(f"Missing item metadata in {split}")
    frame = frame.loc[frame.publish_ts <= cutoff].copy()
    seen = pd.MultiIndex.from_frame(history[["user_id", "note_id"]].drop_duplicates())
    candidate_keys = pd.MultiIndex.from_frame(frame[["user_id", "note_id"]])
    frame = frame.loc[~candidate_keys.isin(seen)].copy()
    if frame.empty:
        raise ValueError(f"No eligible {split} candidates")

    future = read_dataset(data_dir, split)[["user_id", "note_id"]].drop_duplicates()
    available = set(notes.loc[notes.publish_ts <= cutoff, "note_id"])
    future = future.loc[future.note_id.isin(available)].copy()
    keys = pd.MultiIndex.from_frame(future[["user_id", "note_id"]])
    future = future.loc[~keys.isin(seen)].copy()
    truth = future.groupby("user_id")["note_id"].agg(set).to_dict()

    original = (
        frame.sort_values(["user_id", "fine_score", "note_id"], ascending=[True, False, True])
        .groupby("user_id", sort=False).head(20).copy()
    )
    print(f"{split}: cutoff={cutoff}, candidates={len(frame):,}, users={frame.user_id.nunique():,}, "
          f"eligible_future_positives={len(future):,}", flush=True)
    return frame, original, truth, cutoff


def rerank(frame, config, cutoff):
    out = pd.concat(
        (calc_mmr(g, config, cutoff) for _, g in frame.groupby("user_id", sort=True)),
        ignore_index=True
    )
    return out.sort_values(["user_id", "rerank_rank"]).reset_index(drop=True)


def evaluate(frame, truth, k):
    a = accuracy_stats(frame, truth, k)
    d = diversity_stats(frame)
    return {"recall": a["recall"], "ndcg": a["ndcg"],
            "subtopics": d["subtopics"], "ild": d["ild"],
            "categories": d["categories"], "authors": d["authors"]}


def main():
    p = argparse.ArgumentParser(description="V6: select MMR on valid and export final test once")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "raw_v6")
    p.add_argument("--rank-dir", type=Path, default=ROOT / "rank" / "output_v6")
    p.add_argument("--beta-grid", default="0.4")
    p.add_argument("--max-ndcg-drop", type=float, default=0.01,
                   help="Maximum relative NDCG@20 decrease vs fine baseline on valid")
    p.add_argument("--max-recall-drop", type=float, default=0.02,
                   help="Maximum relative Recall@20 decrease vs fine baseline on valid")
    p.add_argument("--topk", type=int, default=20)
    args = p.parse_args()
    if args.topk != 20:
        raise ValueError("This final protocol fixes Top-K to 20")
    if not (0 <= args.max_ndcg_drop < 1 and 0 <= args.max_recall_drop < 1):
        raise ValueError("Maximum drops must be in [0,1)")

    betas = [float(x.strip()) for x in args.beta_grid.split(",") if x.strip()]
    if not betas or not all(np.isfinite(x) and x >= 0 for x in betas):
        raise ValueError("Specify valid nonnegative beta values")
    betas = sorted(set([0.] + betas))  # Always include exact baseline
    print("MMR parameter sweep (VALID ONLY):", betas)
    print("Constraints disabled for K=20; freshness_weight=0, author cap=20")

    notes = read_dataset(args.data, "notes")
    train = read_dataset(args.data, "train")
    valid = read_dataset(args.data, "valid")

    # 1) Select parameters solely using VALID labels.
    vf, vo, vt, vcut = load_stage(args.data, args.rank_dir, notes, train, valid, "valid")
    base = evaluate(vo, vt, 20)
    rows = []
    best = None
    for beta in betas:
        cfg = SimpleNamespace(topk=20, relevance_weight=1., diversity_weight=beta,
                              freshness_weight=0., freshness_half_life=7.,
                              max_per_category=20, max_per_subtopic=20, max_per_author=20)
        if beta == 0:
            score = base.copy()  # Exact zero-reranking baseline
        else:
            out = rerank(vf, cfg, vcut)
            score = evaluate(out, vt, 20)
        feasible = (score["recall"] >= base["recall"] * (1-args.max_recall_drop)
                    and score["ndcg"] >= base["ndcg"] * (1-args.max_ndcg_drop))
        row = {"beta": beta, **score, "feasible": bool(feasible)}
        rows.append(row)
        print(f"VALID beta={beta:0.4f} R@20={score['recall']:.6f} "
              f"NDCG@20={score['ndcg']:.6f} Subtopics@20={score['subtopics']:.4f} "
              f"ILD@20={score['ild']:.4f} feasible={feasible}", flush=True)
        # Among accuracy-constrained feasible models, maximize diversity.
        # Resolve equal diversity using higher NDCG and lower beta.
        ranking = (score["ild"], score["ndcg"], -beta)
        if feasible and (best is None or ranking > best[0]):
            best = (ranking, beta, row)

    beta = best[1] if best else 0.
    print(f"\nFROZEN beta={beta:.4f} chosen from VALID (not TEST)", flush=True)
    args.rank_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.rank_dir / "mmr_v6_valid_sweep.csv", index=False)

    # 2) Test: no tuning and no use of labels for selection.
    tf, to, tt, tcut = load_stage(args.data, args.rank_dir, notes, train, valid, "test")
    cfg = SimpleNamespace(topk=20, relevance_weight=1., diversity_weight=beta,
                          freshness_weight=0., freshness_half_life=7.,
                          max_per_category=20, max_per_subtopic=20, max_per_author=20)
    reranked = rerank(tf, cfg, tcut)
    if reranked.duplicated(["user_id", "note_id"]).any():
        raise AssertionError("Duplicated recommendation pair")
    if int(reranked.groupby("user_id").size().max()) > 20:
        raise AssertionError("More than Top-20 output")
    output = args.rank_dir / "rerank_test_top20_v2.parquet"
    cols = ["user_id", "note_id", "fine_score", "rerank_score", "rerank_rank",
            "category", "subtopic", "author_id", "constraint_relaxation"]
    reranked[cols].to_parquet(output, index=False)

    before = evaluate(to, tt, 20)
    after = evaluate(reranked, tt, 20)
    print("\n========== FINAL TEST (frozen beta) ==========", flush=True)
    print(f"{'Metric':<18} {'FineRank':>12} {'MMR':>12} {'Delta':>12}")
    for key in ("recall", "ndcg", "subtopics", "ild", "categories", "authors"):
        print(f"{key:<18} {before[key]:>12.6f} {after[key]:>12.6f} {after[key]-before[key]:>+12.6f}")
    print(f"Rows={len(reranked):,}, Users={reranked.user_id.nunique():,}")
    print(f"Saved: {output}")
    report = {"selected_beta": beta, "valid_baseline": base, "valid_candidates": rows,
              "test_original": before, "test_mmr": after, "test_rows": len(reranked),
              "test_users": int(reranked.user_id.nunique()),
              "notes": "Parameters selected only with valid; test evaluated after configuration frozen."}
    report_path = args.rank_dir / "rerank_test_top20_v2.metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
