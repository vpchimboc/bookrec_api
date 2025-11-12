# 📚 Book Recommender: Train → Save → Serve via API

This is a minimal, production-friendly template to:
1) **Train** an item-based collaborative filtering recommender
2) **Save** the artifacts to `models/`
3) **Serve** recommendations with **FastAPI**

> Works entirely offline with your CSVs. No external datasets are required.

## Project structure
```
bookrec_api/
├─ app/
│  └─ main.py
├─ data/
│  ├─ ratings.csv
│  └─ books.csv
├─ models/
├─ train.py
└─ requirements.txt
```

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload --port 8000
```

### Try it
```bash
curl http://localhost:8000/health
curl "http://localhost:8000/similar/b1?k=5"
curl -X POST "http://localhost:8000/recommend" -H "Content-Type: application/json" -d '{"user_id": "u1", "k": 5}'
curl "http://localhost:8000/search?q=python"
```

## Data format
- `data/ratings.csv`: `user_id,book_id,rating`
- `data/books.csv`: `book_id,title,author,tags`

## Notes
- Cold-start uses per-item mean rating as a fallback.
- For larger datasets consider approximate nearest neighbors or matrix factorization.
