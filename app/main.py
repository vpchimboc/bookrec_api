from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np
import pandas as pd
import json, os

MODEL_DIR = os.environ.get("MODEL_DIR", "models")

with open(f"{MODEL_DIR}/item_index.json", "r", encoding="utf-8") as f:
    ITEM_INDEX = json.load(f)
with open(f"{MODEL_DIR}/index_item.json", "r", encoding="utf-8") as f:
    INDEX_ITEM = {int(k): v for k, v in json.load(f).items()}
with open(f"{MODEL_DIR}/user_index.json", "r", encoding="utf-8") as f:
    USER_INDEX = json.load(f)
with open(f"{MODEL_DIR}/index_user.json", "r", encoding="utf-8") as f:
    INDEX_USER = {int(k): v for k, v in json.load(f).items()}

ITEM_SIM = np.load(f"{MODEL_DIR}/item_sim.npy")
ITEM_MEANS = np.load(f"{MODEL_DIR}/item_means.npy")
BOOKS = pd.read_parquet(f"{MODEL_DIR}/meta_books.parquet")

user_rated = {}
def build_user_histories():
    ratings_path = os.path.join("data", "ratings.csv")
    if os.path.exists(ratings_path):
        df = pd.read_csv(ratings_path).astype({"user_id": str, "book_id": str})
        for u, grp in df.groupby("user_id"):
            user_rated[u] = set(grp["book_id"].tolist())
build_user_histories()

app = FastAPI(title="Book Recommender API", version="1.0.0")

class RecommendRequest(BaseModel):
    user_id: str
    k: int = 5

def _book_meta(book_id: str):
    row = BOOKS[BOOKS["book_id"] == book_id]
    if row.empty:
        return {"book_id": book_id, "title": None, "author": None, "tags": None}
    return row.iloc[0].to_dict()

@app.get("/health")
def health():
    return {"status": "ok", "items": len(ITEM_INDEX), "users": len(USER_INDEX)}

@app.get("/similar/{book_id}")
def similar_books(book_id: str, k: int = 5):
    if book_id not in ITEM_INDEX:
        raise HTTPException(status_code=404, detail="Unknown book_id")
    idx = ITEM_INDEX[book_id]
    sims = ITEM_SIM[idx]
    top_idx = np.argsort(-sims)[:k]
    out = [{"book_id": INDEX_ITEM[int(i)], "score": float(sims[i]), **_book_meta(INDEX_ITEM[int(i)])} for i in top_idx]
    return {"book_id": book_id, "similar": out}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    user_id = req.user_id
    k = int(req.k)
    if user_id not in USER_INDEX:
        means = ITEM_MEANS.copy()
        top = np.argsort(-means)[:k]
        out = [{"book_id": INDEX_ITEM[int(i)], "score": float(means[i]), **_book_meta(INDEX_ITEM[int(i)])} for i in top]
        return {"user_id": user_id, "strategy": "cold-start-item-means", "items": out}

    rated = user_rated.get(user_id, set())
    if not rated:
        means = ITEM_MEANS.copy()
        top = np.argsort(-means)[:k]
        out = [{"book_id": INDEX_ITEM[int(i)], "score": float(means[i]), **_book_meta(INDEX_ITEM[int(i)])} for i in top]
        return {"user_id": user_id, "strategy": "no-history-item-means", "items": out}

    liked_idx = [ITEM_INDEX[b] for b in rated if b in ITEM_INDEX]
    if not liked_idx:
        means = ITEM_MEANS.copy()
        top = np.argsort(-means)[:k]
        out = [{"book_id": INDEX_ITEM[int(i)], "score": float(means[i]), **_book_meta(INDEX_ITEM[int(i)])} for i in top]
        return {"user_id": user_id, "strategy": "no-liked-found", "items": out}

    agg = ITEM_SIM[:, liked_idx].sum(axis=1)
    for b in rated:
        if b in ITEM_INDEX:
            agg[ITEM_INDEX[b]] = -1e9
    top_idx = np.argsort(-agg)[:k]
    out = [{"book_id": INDEX_ITEM[int(i)], "score": float(agg[i]), **_book_meta(INDEX_ITEM[int(i)])} for i in top_idx]
    return {"user_id": user_id, "strategy": "item-sim-aggregate", "items": out}

@app.get("/search")
def search_books(q: str = Query(..., description="Substring to search in title")):
    hits = BOOKS[BOOKS["title"].str.contains(q, case=False, na=False)].copy()
    return {"count": int(len(hits)), "results": hits.to_dict(orient="records")}
