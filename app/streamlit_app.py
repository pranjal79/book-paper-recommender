"""
streamlit_app.py
────────────────
Streamlit UI for the Book & Research Paper Recommendation System.
Memory-optimized version for Render free tier (512MB RAM).

Changes from v1:
  - TF-IDF loads at startup (small — 2MB)
  - Sentence Transformer loads LAZILY (only when user selects Semantic)
  - gc.collect() after every major operation
  - Metadata text truncated to 500 chars
"""

import os
import sys
import gc
import pickle
import logging
import numpy as np
import pandas as pd
import faiss
import streamlit as st

from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Make sure project root is importable ─────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "models_store").exists():
    ROOT = Path("/app")
sys.path.insert(0, str(ROOT))

from src.models.similarity import query_tfidf, query_semantic
from src.features.preprocessing import clean_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="📚 Book & Paper Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(ROOT, "models_store")

PATHS = {
    "vectorizer":     os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"),
    "tfidf_matrix":   os.path.join(MODELS_DIR, "tfidf_matrix.npz"),
    "embeddings":     os.path.join(MODELS_DIR, "sentence_embeddings.npy"),
    "faiss_tfidf":    os.path.join(MODELS_DIR, "faiss_tfidf.index"),
    "faiss_semantic": os.path.join(MODELS_DIR, "faiss_semantic.index"),
    "metadata":       os.path.join(MODELS_DIR, "metadata.csv"),
}

SOURCE_EMOJI = {"book": "📖", "paper": "🔬"}
METHOD_COLOR = {"tfidf": "#4F8BF9", "semantic": "#F97B4F"}

# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCE LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading TF-IDF model...")
def load_tfidf():
    """Load TF-IDF vectorizer and FAISS index — small, always loaded."""
    with open(PATHS["vectorizer"], "rb") as f:
        vectorizer = pickle.load(f)
    index = faiss.read_index(PATHS["faiss_tfidf"])
    gc.collect()
    logger.info("TF-IDF model loaded successfully")
    return vectorizer, index


@st.cache_resource(show_spinner="Loading Semantic model (first time ~30s)...")
def load_semantic():
    """
    Load Sentence Transformer — LAZY loaded only when user selects Semantic.
    Uses float16 to halve memory usage (~90MB instead of ~180MB).
    """
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # Convert to half precision to save memory
    try:
        model[0].auto_model = model[0].auto_model.half()
        logger.info("Sentence Transformer loaded in float16 mode")
    except Exception:
        logger.info("Sentence Transformer loaded in float32 mode")

    index = faiss.read_index(PATHS["faiss_semantic"])
    gc.collect()
    logger.info("Semantic model loaded successfully")
    return model, index


@st.cache_resource(show_spinner="Loading metadata...")
def load_metadata() -> pd.DataFrame:
    """Load metadata CSV — index maps FAISS result positions to titles."""
    df = pd.read_csv(PATHS["metadata"], index_col="faiss_idx")
    gc.collect()
    return df


def check_artifacts_exist() -> tuple[bool, list[str]]:
    """Check all required model files exist before trying to load."""
    missing = [
        name for name, path in PATHS.items()
        if not os.path.exists(path)
    ]
    return len(missing) == 0, missing


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    .rec-card {
        background: #1E2130;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        border-left: 4px solid #4F8BF9;
        transition: transform 0.15s;
    }
    .rec-card:hover { transform: translateX(3px); }
    .rec-card.semantic { border-left-color: #F97B4F; }

    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-bottom: 4px;
    }
    .card-meta {
        font-size: 0.82rem;
        color: #9BA3B2;
        margin-bottom: 8px;
    }
    .card-preview {
        font-size: 0.88rem;
        color: #C5CAD6;
        line-height: 1.55;
    }
    .score-badge {
        display: inline-block;
        background: #2A3050;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #4F8BF9;
        margin-right: 6px;
    }
    .score-badge.semantic { color: #F97B4F; }
    .source-tag {
        display: inline-block;
        background: #2A3050;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        color: #9BA3B2;
    }
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    hr { border-color: #2A3050 !important; }
    .stTextArea label { font-weight: 600 !important; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT CARD RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_card(row: pd.Series, rank: int, method: str):
    """Render a single recommendation as a styled HTML card."""
    source   = row.get("source", "unknown")
    emoji    = SOURCE_EMOJI.get(source, "📄")
    title    = row.get("title",   "Untitled")
    authors  = str(row.get("authors", "Unknown"))
    category = row.get("category", "—")
    text     = str(row.get("text", ""))
    preview  = text[:280] + "..." if len(text) > 280 else text
    score    = float(row.get("similarity_score", 0))

    card_class  = "rec-card semantic" if method == "semantic" else "rec-card"
    badge_class = "score-badge semantic" if method == "semantic" else "score-badge"

    st.markdown(f"""
    <div class="{card_class}">
        <div class="card-title">#{rank} &nbsp; {emoji} {title}</div>
        <div class="card-meta">
            ✍️ {authors[:80]}
            &nbsp;|&nbsp;
            🏷️ {category}
        </div>
        <div style="margin-bottom:8px;">
            <span class="{badge_class}">Score: {score:.4f}</span>
            <span class="source-tag">{source.upper()}</span>
        </div>
        <div class="card-preview">{preview}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def get_recommendations(
    query: str,
    method: str,
    source_filter: str,
    top_n: int,
    vectorizer,
    tfidf_index,
    sem_model,
    sem_index,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Run query through selected method, apply source filter, return top_n."""
    cleaned_query = clean_text(query, remove_stopwords=True, lemmatize=False)
    if not cleaned_query.strip():
        cleaned_query = query

    fetch_n = top_n * 4

    if method == "TF-IDF":
        results = query_tfidf(
            query_text=cleaned_query,
            vectorizer=vectorizer,
            index=tfidf_index,
            metadata=metadata,
            top_n=fetch_n,
        )
    else:
        results = query_semantic(
            query_text=query,
            model=sem_model,
            index=sem_index,
            metadata=metadata,
            top_n=fetch_n,
        )

    if source_filter == "Books only":
        results = results[results["source"] == "book"]
    elif source_filter == "Papers only":
        results = results[results["source"] == "paper"]

    gc.collect()
    return results.head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar controls and return settings dict."""
    st.sidebar.markdown("## ⚙️ Settings")
    st.sidebar.markdown("---")

    method = st.sidebar.radio(
        "🔍 Recommendation Method",
        options=[
            "TF-IDF (Keyword Match)",
            "Semantic (Best Quality)",
            "Compare Both",
        ],
        index=0,
        help=(
            "**TF-IDF**: Fast keyword-based search — loads instantly.\n\n"
            "**Semantic**: Deep meaning search — loads on first use (~30s).\n\n"
            "**Compare Both**: Shows both side by side."
        ),
    )

    st.sidebar.markdown("---")

    source_filter = st.sidebar.selectbox(
        "📂 Show Results From",
        options=["Books & Papers", "Books only", "Papers only"],
        index=0,
    )

    top_n = st.sidebar.slider(
        "🔢 Number of Recommendations",
        min_value=3,
        max_value=20,
        value=8,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")

    return {
        "method":        method,
        "source_filter": source_filter,
        "top_n":         top_n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE QUERIES
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "A young wizard discovers his magical powers and attends a school of witchcraft",
    "Deep learning methods for natural language processing and text classification",
    "A detective investigates a series of mysterious murders in Victorian London",
    "Reinforcement learning agents that learn to play games from raw pixels",
    "A dystopian society where a totalitarian government controls all information",
    "Graph neural networks for knowledge representation and reasoning",
    "An epic fantasy quest to destroy a powerful dark artifact before evil conquers all",
    "Transformer architecture with self-attention for sequence to sequence tasks",
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    inject_css()

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>📚 Book & Research Paper Recommender</h1>
        <p style="color:#9BA3B2; font-size:1.05rem;">
            Find similar books and research papers using NLP similarity search
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── Check artifacts ───────────────────────────────────────────────────────
    artifacts_ok, missing = check_artifacts_exist()
    if not artifacts_ok:
        st.error(
            f"⚠️ Model artifacts not found: `{', '.join(missing)}`\n\n"
            "Please run the training pipeline first:\n"
            "```bash\n"
            "python src/pipeline/train_pipeline.py --skip-ingestion\n"
            "```"
        )
        st.stop()

    # ── Load lightweight models at startup ────────────────────────────────────
    vectorizer, tfidf_index = load_tfidf()
    metadata                = load_metadata()

    # Semantic model loaded lazily — only when user needs it
    sem_model = None
    sem_index = None

    # ── Sidebar ───────────────────────────────────────────────────────────────
    settings = render_sidebar()

    # Dataset stats in sidebar
    n_books  = (metadata["source"] == "book").sum()
    n_papers = (metadata["source"] == "paper").sum()
    st.sidebar.metric("📖 Books",  f"{n_books:,}")
    st.sidebar.metric("🔬 Papers", f"{n_papers:,}")
    st.sidebar.metric("📄 Total",  f"{len(metadata):,}")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**TF-IDF**: 2k features, bigrams\n\n"
        "**Semantic**: `all-MiniLM-L6-v2`\n\n"
        "**Index**: FAISS FlatIP"
    )

    # ── Memory tip for semantic ───────────────────────────────────────────────
    if "Semantic" in settings["method"] or settings["method"] == "Compare Both":
        st.info(
            "💡 **Semantic search** loads a 90MB model on first use — "
            "expect ~30 seconds on first query. Subsequent queries are instant.",
            icon="ℹ️"
        )

    # ── Query input ───────────────────────────────────────────────────────────
    col_input, col_example = st.columns([3, 1])

    with col_example:
        st.markdown("#### 💡 Try an example")
        selected_example = st.selectbox(
            "Pick an example query",
            options=[""] + EXAMPLE_QUERIES,
            label_visibility="collapsed",
        )

    with col_input:
        st.markdown("#### 🔎 Enter your query")
        default_text = selected_example if selected_example else ""
        query = st.text_area(
            "Enter a book description or paper abstract:",
            value=default_text,
            height=120,
            placeholder=(
                "e.g. 'A young orphan discovers he is a wizard...'\n"
                "   or 'Attention mechanisms for neural machine translation...'"
            ),
            label_visibility="collapsed",
        )

    # ── Search button ─────────────────────────────────────────────────────────
    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        search_clicked = st.button(
            "🔍 Find Similar",
            type="primary",
            use_container_width=True
        )
    with col_clear:
        if st.button("🗑️ Clear"):
            st.rerun()

    st.markdown("---")

    # ── Run recommendations ───────────────────────────────────────────────────
    if search_clicked and query.strip():

        method        = settings["method"]
        source_filter = settings["source_filter"]
        top_n         = settings["top_n"]

        # ── Lazy load semantic if needed ──────────────────────────────────────
        needs_semantic = "Semantic" in method or method == "Compare Both"
        if needs_semantic:
            sem_model, sem_index = load_semantic()

        # ── Single method ─────────────────────────────────────────────────────
        if method != "Compare Both":
            actual_method = "Semantic" if "Semantic" in method else "TF-IDF"
            method_key    = "semantic" if actual_method == "Semantic" else "tfidf"
            color         = METHOD_COLOR[method_key]

            with st.spinner(f"Finding recommendations using {actual_method}..."):
                results = get_recommendations(
                    query=query,
                    method=actual_method,
                    source_filter=source_filter,
                    top_n=top_n,
                    vectorizer=vectorizer,
                    tfidf_index=tfidf_index,
                    sem_model=sem_model,
                    sem_index=sem_index,
                    metadata=metadata,
                )

            if results.empty:
                st.warning("No results found. Try a different query or source filter.")
            else:
                st.markdown(
                    f"### {actual_method} Results "
                    f"<span style='color:{color}; font-size:0.85rem;'>"
                    f"({len(results)} recommendations)</span>",
                    unsafe_allow_html=True,
                )
                for rank, (_, row) in enumerate(results.iterrows(), start=1):
                    render_card(row, rank, method_key)

        # ── Compare Both ──────────────────────────────────────────────────────
        else:
            with st.spinner("Running both methods..."):
                tfidf_results = get_recommendations(
                    query=query,
                    method="TF-IDF",
                    source_filter=source_filter,
                    top_n=top_n,
                    vectorizer=vectorizer,
                    tfidf_index=tfidf_index,
                    sem_model=None,
                    sem_index=None,
                    metadata=metadata,
                )
                sem_results = get_recommendations(
                    query=query,
                    method="Semantic",
                    source_filter=source_filter,
                    top_n=top_n,
                    vectorizer=vectorizer,
                    tfidf_index=tfidf_index,
                    sem_model=sem_model,
                    sem_index=sem_index,
                    metadata=metadata,
                )

            col_tfidf, col_sem = st.columns(2)

            with col_tfidf:
                st.markdown("### 🔵 TF-IDF Results")
                if tfidf_results.empty:
                    st.warning("No TF-IDF results found.")
                else:
                    for rank, (_, row) in enumerate(
                        tfidf_results.iterrows(), start=1
                    ):
                        render_card(row, rank, "tfidf")

            with col_sem:
                st.markdown("### 🟠 Semantic Results")
                if sem_results.empty:
                    st.warning("No semantic results found.")
                else:
                    for rank, (_, row) in enumerate(
                        sem_results.iterrows(), start=1
                    ):
                        render_card(row, rank, "semantic")

    elif search_clicked and not query.strip():
        st.warning("⚠️ Please enter a query before searching.")

    else:
        # ── Landing state ─────────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center; padding:3rem 0; color:#9BA3B2;">
            <div style="font-size:3rem; margin-bottom:1rem;">🔍</div>
            <h3 style="color:#C5CAD6;">How to use this app</h3>
            <p>1. Type a book description or paper abstract in the query box above</p>
            <p>2. Or pick one of the example queries on the right</p>
            <p>3. Choose a recommendation method and filters in the sidebar</p>
            <p>4. Click <strong>Find Similar</strong> to get recommendations</p>
            <br>
            <p style="font-size:0.9rem;">
                💡 <strong>Tip</strong>: Start with
                <strong>TF-IDF</strong> for instant results.
                Switch to <strong>Semantic</strong> for deeper meaning-based search.
            </p>
            <br>
            <p style="font-size:0.9rem;">
                📖 <strong>Books</strong>: CMU Book Summaries &nbsp;|&nbsp;
                🔬 <strong>Papers</strong>: arXiv CS/ML/AI papers
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()