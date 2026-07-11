import os
import logging
import pickle
import numpy as np
import pandas as pd
import faiss
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from src.utils.common import load_params, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf(
    texts: list[str],
    max_features: int,
    ngram_range: tuple,
    save_dir: str,
) -> tuple[TfidfVectorizer, sp.csr_matrix]:
    """
    Fit a TF-IDF vectorizer on the corpus and save:
        - tfidf_vectorizer.pkl  (the fitted vectorizer)
        - tfidf_matrix.npz      (the sparse document-term matrix)
    """
    logger.info(f"Building TF-IDF (max_features={max_features}, ngram={ngram_range})...")

    vectorizer = TfidfVectorizer(
        max_features = max_features,
        ngram_range  = ngram_range,
        sublinear_tf = True,    # apply log(1 + tf) — reduces impact of very frequent terms
        min_df       = 2,       # ignore terms that appear in fewer than 2 documents
        max_df       = 0.95,    # ignore terms that appear in >95% of documents
        strip_accents = "unicode",
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # ── Save vectorizer ──────────────────────────────────────────────────────
    vec_path = os.path.join(save_dir, "tfidf_vectorizer.pkl")
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Saved TF-IDF vectorizer → {vec_path}")

    # ── Save sparse matrix ───────────────────────────────────────────────────
    mat_path = os.path.join(save_dir, "tfidf_matrix.npz")
    sp.save_npz(mat_path, tfidf_matrix)
    logger.info(f"Saved TF-IDF matrix → {mat_path}")

    return vectorizer, tfidf_matrix


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

def build_sentence_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int,
    save_dir: str,
) -> np.ndarray:
    """
    Encode all texts using a Sentence Transformer model and save as .npy.
    Model is downloaded automatically on first run (~90MB for MiniLM).
    """
    logger.info(f"Loading Sentence Transformer model: '{model_name}'...")
    model = SentenceTransformer(model_name)

    logger.info(f"Encoding {len(texts):,} texts in batches of {batch_size}...")
    embeddings = model.encode(
        texts,
        batch_size        = batch_size,
        show_progress_bar = True,
        convert_to_numpy  = True,
        normalize_embeddings = True,   # L2-normalize → cosine sim = dot product (faster FAISS)
    )

    logger.info(f"Embeddings shape: {embeddings.shape}")  # (n_docs, 384)

    emb_path = os.path.join(save_dir, "sentence_embeddings.npy")
    np.save(emb_path, embeddings)
    logger.info(f"Saved embeddings → {emb_path}")

    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# FAISS INDEXING
# ─────────────────────────────────────────────────────────────────────────────

def build_faiss_index_tfidf(
    tfidf_matrix: sp.csr_matrix,
    save_dir: str,
    nlist: int,
) -> faiss.Index:
    """
    Build a FAISS index over the TF-IDF dense projection.
    TF-IDF is sparse, so we convert to dense float32 first.

    For our scale (~10k docs) we use a flat L2 index (exact search).
    For >100k docs, switch to IndexIVFFlat for speed.
    """
    logger.info("Building FAISS index for TF-IDF vectors...")

    # Convert sparse → dense float32
    dense = tfidf_matrix.toarray().astype("float32")

    # L2-normalize so dot product = cosine similarity
    faiss.normalize_L2(dense)

    dim = dense.shape[1]
    logger.info(f"TF-IDF dense shape: {dense.shape} (dim={dim})")

    if dense.shape[0] < nlist * 39:
        # Not enough data for IVF — fall back to exact flat index
        logger.info("Using IndexFlatIP (exact cosine search) for TF-IDF.")
        index = faiss.IndexFlatIP(dim)
    else:
        logger.info(f"Using IndexIVFFlat (nlist={nlist}) for TF-IDF.")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(dense)

    index.add(dense)
    logger.info(f"FAISS TF-IDF index: {index.ntotal} vectors indexed")

    idx_path = os.path.join(save_dir, "faiss_tfidf.index")
    faiss.write_index(index, idx_path)
    logger.info(f"Saved FAISS TF-IDF index → {idx_path}")

    return index


def build_faiss_index_semantic(
    embeddings: np.ndarray,
    save_dir: str,
    nlist: int,
) -> faiss.Index:
    """
    Build a FAISS index for sentence embeddings.
    Embeddings are already L2-normalized (done in build_sentence_embeddings),
    so inner product = cosine similarity.
    """
    logger.info("Building FAISS index for sentence embeddings...")

    vectors = embeddings.astype("float32")
    dim = vectors.shape[1]

    if vectors.shape[0] < nlist * 39:
        logger.info("Using IndexFlatIP (exact cosine search) for embeddings.")
        index = faiss.IndexFlatIP(dim)
    else:
        logger.info(f"Using IndexIVFFlat (nlist={nlist}) for embeddings.")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)

    index.add(vectors)
    logger.info(f"FAISS Semantic index: {index.ntotal} vectors indexed")

    idx_path = os.path.join(save_dir, "faiss_semantic.index")
    faiss.write_index(index, idx_path)
    logger.info(f"Saved FAISS Semantic index → {idx_path}")

    return index


# ─────────────────────────────────────────────────────────────────────────────
# METADATA STORE
# ─────────────────────────────────────────────────────────────────────────────

def save_metadata(df: pd.DataFrame, save_dir: str):
    """
    Save a lightweight metadata CSV that the recommender uses to
    look up titles/authors/categories by FAISS result index.
    FAISS returns row indices — we map those back to human-readable info here.
    """
    meta_cols = ["item_id", "source", "title", "authors", "category", "text"]
    meta = df[meta_cols].reset_index(drop=True)
    meta.index.name = "faiss_idx"   # faiss_idx == row position in the index

    meta_path = os.path.join(save_dir, "metadata.csv")
    meta.to_csv(meta_path)
    logger.info(f"Saved metadata ({len(meta):,} rows) → {meta_path}")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# QUERY FUNCTIONS  (used later by the Streamlit app & pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def query_tfidf(
    query_text: str,
    vectorizer: TfidfVectorizer,
    index: faiss.Index,
    metadata: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return top-N similar items for a raw query string using TF-IDF."""
    vec = vectorizer.transform([query_text]).toarray().astype("float32")
    faiss.normalize_L2(vec)
    scores, indices = index.search(vec, top_n + 1)  # +1 in case query itself is in corpus

    results = metadata.iloc[indices[0]].copy()
    results["similarity_score"] = scores[0]
    results["method"] = "tfidf"
    return results[results["similarity_score"] < 0.9999].head(top_n)  # exclude self-match


def query_semantic(
    query_text: str,
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return top-N similar items using Sentence Transformer embeddings."""
    vec = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, indices = index.search(vec, top_n + 1)

    results = metadata.iloc[indices[0]].copy()
    results["similarity_score"] = scores[0]
    results["method"] = "semantic"
    return results[results["similarity_score"] < 0.9999].head(top_n)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_extraction():
    all_params = load_params()
    params     = all_params["features"]

    save_dir = params["models_store_dir"]
    ensure_dir(save_dir)

    # ── Load processed data ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 1/5 — LOADING PROCESSED DATA")
    logger.info("=" * 55)

    df = pd.read_csv(params["processed_data_path"])
    df = df.dropna(subset=[params["text_column"]])
    df = df[df[params["text_column"]].str.strip().str.len() > 0].reset_index(drop=True)

    texts = df[params["text_column"]].tolist()
    logger.info(f"Loaded {len(texts):,} documents for feature extraction")

    # ── TF-IDF ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 2/5 — TF-IDF VECTORIZATION")
    logger.info("=" * 55)

    vectorizer, tfidf_matrix = build_tfidf(
        texts        = texts,
        max_features = params["tfidf_max_features"],
        ngram_range  = (params["tfidf_ngram_min"], params["tfidf_ngram_max"]),
        save_dir     = save_dir,
    )

    # ── Sentence Embeddings ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 3/5 — SENTENCE EMBEDDINGS")
    logger.info("=" * 55)

    embeddings = build_sentence_embeddings(
        texts      = texts,
        model_name = params["embedding_model_name"],
        batch_size = params["embedding_batch_size"],
        save_dir   = save_dir,
    )

    # ── FAISS Indexing ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 4/5 — BUILDING FAISS INDICES")
    logger.info("=" * 55)

    build_faiss_index_tfidf(
        tfidf_matrix = tfidf_matrix,
        save_dir     = save_dir,
        nlist        = params["faiss_nlist"],
    )

    build_faiss_index_semantic(
        embeddings = embeddings,
        save_dir   = save_dir,
        nlist      = params["faiss_nlist"],
    )

    # ── Metadata ─────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 5/5 — SAVING METADATA")
    logger.info("=" * 55)

    save_metadata(df, save_dir)

    logger.info("\n" + "=" * 55)
    logger.info("  FEATURE EXTRACTION COMPLETE ✅")
    logger.info("=" * 55)


if __name__ == "__main__":
    run_feature_extraction()