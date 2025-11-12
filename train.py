# Train an item-based collaborative filtering recommender for books.
# Inputs:
# - data/ratings.csv: columns user_id, book_id, rating
# - data/books.csv: columns book_id, title, author, tags
# Outputs (in models/):
# - item_index.json, index_item.json
# - user_index.json, index_user.json
# - item_sim.npy, item_means.npy
# - meta_books.parquet

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.environ.get("DATA_DIR", "data")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

def load_data():
    ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
    books = pd.read_csv(f"{DATA_DIR}/books.csv")
    for col in ["user_id", "book_id", "rating"]:
        if col not in ratings.columns:
            raise ValueError(f"ratings.csv must include column {col}")
    for col in ["book_id", "title"]:
        if col not in books.columns:
            raise ValueError(f"books.csv must include column {col}")
    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["book_id"] = ratings["book_id"].astype(str)
    return ratings, books

def build_mappings(ratings):
    users = ratings["user_id"].astype(str).unique()
    items = ratings["book_id"].astype(str).unique()
    user_to_idx = {u:i for i,u in enumerate(sorted(users))}
    idx_to_user = {i:u for u,i in user_to_idx.items()}
    item_to_idx = {it:i for i,it in enumerate(sorted(items))}
    idx_to_item = {i:it for it,i in item_to_idx.items()}
    return user_to_idx, idx_to_user, item_to_idx, idx_to_item

def build_user_item_matrix(ratings, user_to_idx, item_to_idx):
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    mat = np.zeros((n_users, n_items), dtype=np.float32)
    counts = np.zeros((n_users, n_items), dtype=np.int32)
    for _, row in ratings.iterrows():
        ui = user_to_idx[str(row["user_id"])]
        ii = item_to_idx[str(row["book_id"])]
        mat[ui, ii] += float(row["rating"])
        counts[ui, ii] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.where(counts > 0, mat / np.maximum(counts, 1), 0.0)
    return mat

def compute_item_similarity(user_item):
    # mean-center each item (column)
    counts = (user_item != 0).sum(axis=0)
    sums = user_item.sum(axis=0)
    means = np.where(counts>0, sums / np.maximum(counts,1), 0.0)
    centered = user_item - means
    centered[:, counts==0] = 0.0
    sim = cosine_similarity(centered.T)
    np.fill_diagonal(sim, 0.0)
    return sim, means

def save_artifacts(user_to_idx, idx_to_user, item_to_idx, idx_to_item, sim, item_means, books):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f"{MODEL_DIR}/user_index.json", "w", encoding="utf-8") as f:
        json.dump(user_to_idx, f, ensure_ascii=False)
    with open(f"{MODEL_DIR}/index_user.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_user, f, ensure_ascii=False)
    with open(f"{MODEL_DIR}/item_index.json", "w", encoding="utf-8") as f:
        json.dump(item_to_idx, f, ensure_ascii=False)
    with open(f"{MODEL_DIR}/index_item.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_item, f, ensure_ascii=False)
    np.save(f"{MODEL_DIR}/item_sim.npy", sim.astype(np.float32))
    np.save(f"{MODEL_DIR}/item_means.npy", item_means.astype(np.float32))
    books.drop_duplicates(subset=["book_id"]).to_parquet(f"{MODEL_DIR}/meta_books.parquet", index=False)

def main():
    ratings, books = load_data()
    user_to_idx, idx_to_user, item_to_idx, idx_to_item = build_mappings(ratings)
    ui = build_user_item_matrix(ratings, user_to_idx, item_to_idx)
    sim, item_means = compute_item_similarity(ui)
    save_artifacts(user_to_idx, idx_to_user, item_to_idx, idx_to_item, sim, item_means, books)
    print("✅ Training complete. Artifacts saved to:", MODEL_DIR)
    print("Users:", len(user_to_idx), "Items:", len(item_to_idx))

if __name__ == "__main__":
    main()
