from pathlib import Path
import math

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


DATA_DIR = Path("data/raw_v6")
# serve/api.py 第 12 行左右
RERANK_PATH = Path("rank/output_v6/rerank_test_top20_v2.parquet")
NOTES_PATH = DATA_DIR / "notes.parquet"

app = FastAPI(title="XHS RecSys API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: dict = {}


def read_any(p: Path) -> pd.DataFrame:
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unknown format: {p}")


def load_rerank() -> pd.DataFrame:
    if "rerank" not in _cache:
        if not RERANK_PATH.exists():
            raise FileNotFoundError(f"缺少重排输出: {RERANK_PATH}")
        df = read_any(RERANK_PATH)
        if "rerank_rank" not in df.columns:
            df = df.sort_values(["user_id", "rerank_score"], ascending=[True, False])
            df["rerank_rank"] = df.groupby("user_id").cumcount() + 1
        _cache["rerank"] = df
    return _cache["rerank"]


def load_notes() -> pd.DataFrame:
    if "notes" not in _cache:
        if not NOTES_PATH.exists():
            raise FileNotFoundError(f"缺少 notes: {NOTES_PATH}")
        _cache["notes"] = read_any(NOTES_PATH)
    return _cache["notes"]


def clean_nan(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


@app.get("/health")
def health():
    return {"ok": True, "rerank_exists": RERANK_PATH.exists()}


@app.get("/users")
def users(limit: int = Query(500, ge=1, le=10000)):
    df = load_rerank()
    ids = sorted(df["user_id"].unique().tolist())
    return {
        "total": int(df["user_id"].nunique()),
        "users": ids[:limit],
    }


@app.get("/recommend/{user_id}")
def recommend(user_id: int, topk: int = Query(20, ge=1, le=100)):
    df = load_rerank()
    sub = df[df["user_id"] == user_id].sort_values("rerank_rank").head(topk).copy()
    if sub.empty:
        raise HTTPException(404, f"user {user_id} 无推荐")

    # 在当前候选内按 fine_score 排个名次
    sub["fine_rank"] = sub["fine_score"].rank(ascending=False, method="first").astype(int)

    notes = load_notes()
    keep = [c for c in ["note_id", "title", "tags", "publish_ts", "is_cold"] if c in notes.columns]
    sub = sub.merge(notes[keep], on="note_id", how="left")

    summary = {
        "n_categories": int(sub["category"].nunique()) if "category" in sub.columns else 0,
        "n_authors": int(sub["author_id"].nunique()) if "author_id" in sub.columns else 0,
        "category_dist": sub["category"].value_counts().to_dict() if "category" in sub.columns else {},
    }

    return {
        "user_id": user_id,
        "topk": topk,
        "summary": summary,
        "items": [clean_nan(r) for r in sub.to_dict(orient="records")],
    }