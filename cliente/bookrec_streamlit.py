
import os
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="📚 Recomendador de Libros", page_icon="📚", layout="wide")

st.title("📚 Recomendador de Libros — Streamlit")

st.sidebar.header("⚙️ Configuración")

mode = st.sidebar.radio("Modo de uso", ["Usar API FastAPI", "Cargar artefactos locales"], index=0)
k = st.sidebar.slider("Cantidad de recomendaciones (k)", 1, 20, 5)

if mode == "Usar API FastAPI":
    base_url = st.sidebar.text_input("URL de la API", value="http://localhost:8000")
else:
    model_dir = st.sidebar.text_input("Carpeta de modelos", value="models")
    data_dir = st.sidebar.text_input("Carpeta de datos (para historial de usuario)", value="data")

st.sidebar.markdown("---")
st.sidebar.caption("Sugerencia: en 'Usar API', corre previamente: `uvicorn app.main:app --reload --port 8000`")

# ---------- Helpers API ----------
def api_health(base_url: str):
    r = requests.get(f"{base_url.rstrip('/')}/health", timeout=10)
    r.raise_for_status()
    return r.json()

def api_recommend(base_url: str, user_id: str, k: int):
    payload = {"user_id": user_id, "k": k}
    r = requests.post(f"{base_url.rstrip('/')}/recommend", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

def api_similar(base_url: str, book_id: str, k: int):
    r = requests.get(f"{base_url.rstrip('/')}/similar/{book_id}", params={"k": k}, timeout=10)
    r.raise_for_status()
    return r.json()

def api_search(base_url: str, q: str):
    r = requests.get(f"{base_url.rstrip('/')}/search", params={"q": q}, timeout=10)
    r.raise_for_status()
    return r.json()

# ---------- Helpers Local Model ----------
@st.cache_resource(show_spinner=False)
def load_local_artifacts(model_dir: str):
    item_index = json.load(open(os.path.join(model_dir, "item_index.json"), "r", encoding="utf-8"))
    index_item = {int(k): v for k, v in json.load(open(os.path.join(model_dir, "index_item.json"), "r", encoding="utf-8")).items()}
    item_sim = np.load(os.path.join(model_dir, "item_sim.npy"))
    item_means = np.load(os.path.join(model_dir, "item_means.npy"))
    books = pd.read_parquet(os.path.join(model_dir, "meta_books.parquet"))
    return item_index, index_item, item_sim, item_means, books

@st.cache_resource(show_spinner=False)
def load_user_hist(data_dir: str):
    path = os.path.join(data_dir, "ratings.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path).astype({"user_id": str, "book_id": str})
    return {u: set(g["book_id"]) for u, g in df.groupby("user_id")}

def local_book_meta(books_df: pd.DataFrame, book_id: str):
    row = books_df[books_df["book_id"] == book_id]
    return row.iloc[0].to_dict() if not row.empty else {"book_id": book_id, "title": None, "author": None, "tags": None}

def local_recommend(user_id: str, k: int, model_dir: str, data_dir: str):
    item_index, index_item, item_sim, item_means, books = load_local_artifacts(model_dir)
    user_hist = load_user_hist(data_dir)
    rated = user_hist.get(user_id, set())

    if not rated:
        top = np.argsort(-item_means)[:k]
        return {
            "user_id": user_id,
            "strategy": "cold-start-item-means",
            "items": [
                {"book_id": index_item[int(i)], "score": float(item_means[i]), **local_book_meta(books, index_item[int(i)])}
                for i in top
            ],
        }

    liked_idx = [item_index[b] for b in rated if b in item_index]
    if not liked_idx:
        top = np.argsort(-item_means)[:k]
        return {
            "user_id": user_id,
            "strategy": "no-liked-found",
            "items": [
                {"book_id": index_item[int(i)], "score": float(item_means[i]), **local_book_meta(books, index_item[int(i)])}
                for i in top
            ],
        }

    agg = item_sim[:, liked_idx].sum(axis=1)
    for b in rated:
        if b in item_index:
            agg[item_index[b]] = -1e9
    top_idx = np.argsort(-agg)[:k]
    return {
        "user_id": user_id,
        "strategy": "item-sim-aggregate",
        "items": [
            {"book_id": index_item[int(i)], "score": float(agg[i]), **local_book_meta(books, index_item[int(i)])}
            for i in top_idx
        ],
    }

def local_similar(book_id: str, k: int, model_dir: str):
    item_index, index_item, item_sim, item_means, books = load_local_artifacts(model_dir)
    if book_id not in item_index:
        return {"book_id": book_id, "similar": []}
    idx = item_index[book_id]
    sims = item_sim[idx]
    top_idx = np.argsort(-sims)[:k]
    return {
        "book_id": book_id,
        "similar": [
            {"book_id": index_item[int(i)], "score": float(sims[i]), **local_book_meta(books, index_item[int(i)])}
            for i in top_idx
        ],
    }

def local_search(q: str, model_dir: str):
    _, _, _, _, books = load_local_artifacts(model_dir)
    hits = books[books["title"].str.contains(q, case=False, na=False)].copy()
    return {"count": int(len(hits)), "results": hits.to_dict(orient="records")}

# ---------- UI Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["✅ Salud / Estado", "⭐ Recomendaciones", "📚 Similares", "🔎 Búsqueda"])

with tab1:
    st.subheader("Estado del servicio / artefactos")
    if mode == "Usar API FastAPI":
        try:
            info = api_health(base_url)
            st.success("API OK")
            st.json(info)
        except Exception as e:
            st.error(f"No se pudo conectar a la API: {e}")
    else:
        try:
            item_index, index_item, item_sim, item_means, books = load_local_artifacts(model_dir)
            st.success("Artefactos locales cargados")
            st.write(f"Items: {len(item_index)} | Ejemplos de libros:")
            st.dataframe(books.head(10))
        except Exception as e:
            st.error(f"Error cargando artefactos locales: {e}")

with tab2:
    st.subheader("Recomendaciones por usuario")
    colu1, colu2 = st.columns([1,1])
    with colu1:
        user_id = st.text_input("user_id", value="u1")
    with colu2:
        k_val = st.number_input("k", min_value=1, max_value=50, value=k)

    if st.button("Recomendar"):
        try:
            if mode == "Usar API FastAPI":
                res = api_recommend(base_url, user_id, int(k_val))
                items = res.get("items", [])
            else:
                res = local_recommend(user_id, int(k_val), model_dir, data_dir)
                items = res.get("items", [])
            if not items:
                st.warning("No hay recomendaciones disponibles.")
            else:
                for it in items:
                    with st.container(border=True):
                        st.markdown(f"**{it.get('title') or it['book_id']}**")
                        st.caption(f"Autor: {it.get('author') or 'Desconocido'} | ID: `{it['book_id']}` | Score: {it.get('score'):.3f}")
                        if it.get("tags"):
                            st.write("Etiquetas:", it["tags"])
        except Exception as e:
            st.error(f"Error al recomendar: {e}")

with tab3:
    st.subheader("Libros similares a un `book_id`")
    col1, col2 = st.columns([1,1])
    with col1:
        book_id = st.text_input("book_id", value="b3")
    with col2:
        k_val2 = st.number_input("k similares", min_value=1, max_value=50, value=k)

    if st.button("Buscar similares"):
        try:
            if mode == "Usar API FastAPI":
                res = api_similar(base_url, book_id, int(k_val2))
                items = res.get("similar", [])
            else:
                res = local_similar(book_id, int(k_val2), model_dir)
                items = res.get("similar", [])
            if not items:
                st.warning("No se encontraron similares.")
            else:
                for it in items:
                    with st.container(border=True):
                        st.markdown(f"**{it.get('title') or it['book_id']}**")
                        st.caption(f"Autor: {it.get('author') or 'Desconocido'} | ID: `{it['book_id']}` | Sim: {it.get('score'):.3f}")
                        if it.get("tags"):
                            st.write("Etiquetas:", it["tags"])
        except Exception as e:
            st.error(f"Error al buscar similares: {e}")

with tab4:
    st.subheader("Búsqueda por título")
    query = st.text_input("Consulta", value="python")
    if st.button("Buscar"):
        try:
            if mode == "Usar API FastAPI":
                res = api_search(base_url, query)
                results = res.get("results", [])
            else:
                res = local_search(query, model_dir)
                results = res.get("results", [])
            if not results:
                st.info("Sin resultados.")
            else:
                st.write(f"Resultados: {len(results)}")
                for it in results:
                    with st.container(border=True):
                        st.markdown(f"**{it.get('title') or it['book_id']}**")
                        st.caption(f"Autor: {it.get('author') or 'Desconocido'} | ID: `{it['book_id']}`")
                        if it.get("tags"):
                            st.write("Etiquetas:", it["tags"])
        except Exception as e:
            st.error(f"Error en la búsqueda: {e}")
