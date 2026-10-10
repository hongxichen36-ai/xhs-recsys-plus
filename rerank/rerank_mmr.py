"""V6 topic-aware MMR reranking of Residual DeepFM v2 candidates.

Put at: xhs-recsys/rerank/rerank_mmr_v6.py
Direct run (VALID first): python rerank/rerank_mmr_v6.py
Final test after fixing params: python rerank/rerank_mmr_v6.py --split test --final-test

Ranking never uses validation/test interactions; they are read only for reporting.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_table(root, stem):
    for suffix in (".parquet", ".csv"):
        path = Path(root) / f"{stem}{suffix}"
        if path.exists():
            return pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
    raise FileNotFoundError(f"Missing {stem}.parquet/.csv in {root}")


def calc_mmr(group, cfg, cutoff):
    """Select Top-K with incremental O(K*N) max-redundancy updates."""
    group = group.sort_values(["fine_score", "note_id"], ascending=[False, True]).copy()
    scores = group["fine_score"].to_numpy("float64")
    lo, hi = scores.min(), scores.max()
    rel = ((scores - lo) / (hi - lo)) if hi > lo else np.full(len(scores), 0.5)
    ages = np.maximum(0, (cutoff - group["publish_ts"].to_numpy("int64")) / 86400.0)
    fresh = np.exp(-np.log(2) * ages / cfg.freshness_half_life)

    cats = group["category"].astype(str).to_numpy()
    # 'subtopic' is the real micro-topic field in xhs_topk_v6.py.
    topics = (group["category"].astype(str) + "::" + group["subtopic"].astype(str)).to_numpy()
    authors = group["author_id"].to_numpy("int64")
    remain = np.ones(len(group), dtype=bool)
    redundancy = np.zeros(len(group), dtype="float64")
    selected, values, relax_levels = [], [], []
    cat_count, topic_count, author_count = {}, {}, {}

    for _ in range(min(cfg.topk, len(group))):
        objective = cfg.relevance_weight * rel + cfg.freshness_weight * fresh - cfg.diversity_weight * redundancy
        cat_ok = np.fromiter((cat_count.get(v, 0) < cfg.max_per_category for v in cats), dtype=bool, count=len(cats))
        topic_ok = np.fromiter((topic_count.get(v, 0) < cfg.max_per_subtopic for v in topics), dtype=bool, count=len(topics))
        author_ok = np.fromiter((author_count.get(int(v), 0) < cfg.max_per_author for v in authors), dtype=bool, count=len(authors))
        # Relax constraints only if needed to keep a complete Top-K.
        masks = [
            remain & cat_ok & topic_ok & author_ok,
            remain & cat_ok & author_ok,
            remain & author_ok,
            remain,
        ]
        for level, mask in enumerate(masks):
            if mask.any():
                break
        choices = np.flatnonzero(mask)
        # Stable tie breaking follows the original fine-score order.
        picked = choices[np.argmax(objective[choices])]
        selected.append(int(picked))
        values.append(float(objective[picked]))
        relax_levels.append(level)
        remain[picked] = False
        cat = cats[picked]
        topic = topics[picked]
        author = int(authors[picked])
        cat_count[cat] = cat_count.get(cat, 0) + 1
        topic_count[topic] = topic_count.get(topic, 0) + 1
        author_count[author] = author_count.get(author, 0) + 1
        # V6 tags are precisely the subtopic. Avoid double-counting tags.
        similarity = 0.2 * (cats == cat) + 0.8 * (topics == topic)
        redundancy = np.maximum(redundancy, similarity)

    out = group.iloc[selected].copy()
    out["rerank_rank"] = np.arange(1, len(out) + 1, dtype=np.int16)
    out["rerank_score"] = np.asarray(values, dtype=np.float32)
    out["constraint_relaxation"] = np.asarray(relax_levels, dtype=np.int8)
    return out


def diversity_stats(frame):
    """All summary metrics are user-macro averages."""
    result = []
    for _, g in frame.groupby("user_id", sort=False):
        cat = g["category"].astype(str).to_numpy()
        topic = (g["category"].astype(str) + "::" + g["subtopic"].astype(str)).to_numpy()
        n = len(g)
        if n < 2:
            ild = 0.0
        else:
            # The same 0.2/0.8 content similarity as the MMR objective.
            same_cat = (cat[:, None] == cat[None, :])
            same_topic = (topic[:, None] == topic[None, :])
            tri = np.triu_indices(n, k=1)
            ild = float((1 - 0.2 * same_cat[tri] - 0.8 * same_topic[tri]).mean())
        result.append((len(set(cat)), len(set(topic)), g["author_id"].nunique(), ild))
    if not result:
        return dict(categories=0., subtopics=0., authors=0., ild=0.)
    avg = np.asarray(result, dtype="float64").mean(axis=0)
    return dict(zip(("categories", "subtopics", "authors", "ild"), map(float, avg)))


def accuracy_stats(frame, truth, topk):
    """Recall/NDCG uses the full future eligible positive set as denominator."""
    lists = frame.groupby("user_id")["note_id"].agg(list).to_dict()
    recalls, ndcgs = [], []
    for uid, gt in truth.items():
        ids = lists.get(uid, [])[:topk]
        hits = np.asarray([int(x in gt) for x in ids], dtype=float)
        recalls.append(float(hits.sum() / len(gt)))
        dcg = float((hits / np.log2(np.arange(len(ids)) + 2)).sum())
        ideal = float((1 / np.log2(np.arange(min(topk, len(gt))) + 2)).sum())
        ndcgs.append(dcg / ideal if ideal else 0.)
    return dict(recall=float(np.mean(recalls)) if recalls else 0.,
                ndcg=float(np.mean(ndcgs)) if ndcgs else 0., users=len(truth))


def main():
    p = argparse.ArgumentParser(description="V6 topic-aware MMR fine-rank reranking")
    p.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "raw_v6")
    p.add_argument("--rank-dir", type=Path, default=PROJECT_ROOT / "rank" / "output_v6")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--split", choices=("valid", "test"), default="valid")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--diversity-weight", type=float, default=0.02)
    p.add_argument("--relevance-weight", type=float, default=1.0)
    p.add_argument("--freshness-weight", type=float, default=0.01)
    p.add_argument("--freshness-half-life", type=float, default=7.0)
    # Avoid v5's category cap=5, which conflicts with V6 focused interests.
    p.add_argument("--max-per-category", type=int, default=20)
    p.add_argument("--max-per-subtopic", type=int, default=16)
    p.add_argument("--max-per-author", type=int, default=2)
    p.add_argument("--final-test", action="store_true", help="Print TEST metrics only for the final frozen configuration")
    args = p.parse_args()

    if min(args.topk, args.max_per_category, args.max_per_subtopic,
           args.max_per_author, args.freshness_half_life) <= 0:
        raise ValueError("TopK, caps and half life must be positive")
    if min(args.diversity_weight, args.relevance_weight, args.freshness_weight) < 0:
        raise ValueError("Weights must be nonnegative")
    src = args.input or args.rank_dir / f"fine_{args.split}_top200_v2.parquet"
    dest = args.output or args.rank_dir / f"rerank_{args.split}_top{args.topk}_v2.parquet"
    if not src.exists():
        raise FileNotFoundError(f"DeepFM output missing: {src}")
    notes = read_table(args.data, "notes")
    train = read_table(args.data, "train")
    history = train if args.split == "valid" else pd.concat([train, read_table(args.data, "valid")], ignore_index=True)
    cutoff = int(history["timestamp"].max())
    pred = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    needed = {"user_id", "note_id", "fine_score"}
    if not needed.issubset(pred.columns):
        raise ValueError(f"Fine result missing fields: {sorted(needed - set(pred.columns))}")
    if pred.duplicated(["user_id", "note_id"]).any():
        raise ValueError("Fine rank contains repeated user-note pairs")
    if not np.isfinite(pred.fine_score).all():
        raise ValueError("fine_score contains NaN or infinity")
    meta_cols = ["note_id", "category", "subtopic", "author_id", "publish_ts"]
    if not set(meta_cols).issubset(notes.columns):
        raise ValueError("V6 notes requires category/subtopic/author_id/publish_ts")
    pred = pred.merge(notes[meta_cols], on="note_id", how="left", validate="many_to_one")
    if pred[meta_cols[1:]].isna().any().any():
        raise ValueError("Note metadata missing")
    before = len(pred)
    pred = pred.loc[pred.publish_ts <= cutoff].copy()
    history_pairs = pd.MultiIndex.from_frame(history[["user_id", "note_id"]].drop_duplicates())
    current_pairs = pd.MultiIndex.from_frame(pred[["user_id", "note_id"]])
    pred = pred.loc[~current_pairs.isin(history_pairs)].copy()
    if pred.empty:
        raise ValueError("All candidates were filtered")
    print(f"Split={args.split}, cutoff={cutoff}; candidates={before:,} -> eligible={len(pred):,}", flush=True)

    original = (pred.sort_values(["user_id", "fine_score", "note_id"], ascending=[True, False, True])
                .groupby("user_id", sort=False).head(args.topk).copy())
    reranked = pd.concat([calc_mmr(g, args, cutoff) for _, g in pred.groupby("user_id", sort=True)], ignore_index=True)
    reranked = reranked.sort_values(["user_id", "rerank_rank"]).reset_index(drop=True)
    if reranked.duplicated(["user_id", "note_id"]).any():
        raise AssertionError("Repeated recommendation pair")
    dest.parent.mkdir(parents=True, exist_ok=True)
    outcols = ["user_id", "note_id", "fine_score", "rerank_score", "rerank_rank",
               "category", "subtopic", "author_id", "constraint_relaxation"]
    if dest.suffix == ".csv":
        reranked[outcols].to_csv(dest, index=False, encoding="utf-8-sig")
    else:
        reranked[outcols].to_parquet(dest, index=False)
    print(f"Exported {len(reranked):,} rows for {reranked.user_id.nunique():,} users -> {dest}")
    if (reranked.constraint_relaxation > 0).any():
        print(f"Constraint relaxations: {(reranked.constraint_relaxation > 0).sum():,} rows")

    base_d = diversity_stats(original)
    new_d = diversity_stats(reranked)
    print("\nDiversity (user averages):")
    print(f"{'Metric':<22}{'FineRank':>12}{'MMR':>12}{'Delta':>12}")
    labels = (("categories", "Categories@20"), ("subtopics", "Subtopics@20"),
              ("authors", "Authors@20"), ("ild", "ILD@20"))
    for key, label in labels:
        print(f"{label:<22}{base_d[key]:>12.4f}{new_d[key]:>12.4f}{new_d[key]-base_d[key]:>+12.4f}")

    report = {"split": args.split, "cutoff": cutoff, "topk": args.topk,
              "diversity_weight": args.diversity_weight, "fine_diversity": base_d,
              "mmr_diversity": new_d}
    if args.split == "valid" or args.final_test:
        target = read_table(args.data, args.split)[["user_id", "note_id"]].drop_duplicates()
        eligible_ids = set(notes.loc[notes.publish_ts <= cutoff, "note_id"])
        target = target.loc[target.note_id.isin(eligible_ids)].copy()
        current = pd.MultiIndex.from_frame(target[["user_id", "note_id"]])
        target = target.loc[~current.isin(history_pairs)].copy()
        truth = target.groupby("user_id")["note_id"].agg(set).to_dict()
        basem = accuracy_stats(original, truth, args.topk)
        newm = accuracy_stats(reranked, truth, args.topk)
        print("\nAccuracy (same full positive ground truth):")
        print(f"FineRank Recall@{args.topk}={basem['recall']:.6f} NDCG@{args.topk}={basem['ndcg']:.6f}")
        print(f"MMR      Recall@{args.topk}={newm['recall']:.6f} NDCG@{args.topk}={newm['ndcg']:.6f}")
        report.update(fine_accuracy=basem, mmr_accuracy=newm)
    else:
        print("\nTest accuracy suppressed. Pass --final-test ONCE after fixing parameters on VALID.")
    metrics_path = dest.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Metrics saved:", metrics_path)


if __name__ == "__main__":
    main()
