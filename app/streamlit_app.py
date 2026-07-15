"""
streamlit_app.py
────────────────
Streamlit UI for the Book & Research Paper Recommendation System.

Features:
  - TF-IDF keyword-based recommendations
  - Semantic (Sentence Transformer) recommendations
  - Side-by-side comparison mode
  - Filter by source (books / papers / both)
  - Adjustable top-N results
  - Result cards with similarity scores
"""

import os
import sys
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
    "vectorizer":      os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"),
    "tfidf_matrix":    os.path.join(MODELS_DIR, "tfidf_matrix.npz"),
    "embeddings":      os.path.join(MODELS_DIR, "sentence_embeddings.npy"),
    "faiss_tfidf":     os.path.join(MODELS_DIR, "faiss_tfidf.index"),
    "faiss_semantic":  os.path.join(MODELS_DIR, "faiss_semantic.index"),
    "metadata":        os.path.join(MODELS_DIR, "metadata.csv"),
}

SOURCE_EMOJI = {"book": "📖", "paper": "🔬"}
METHOD_COLOR = {"tfidf": "#4F8BF9", "semantic": "#F97B4F"}

# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCE LOADERS
# Runs only ONCE per app session — not on every interaction
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading TF-IDF model...")
def load_tfidf():
    with open(PATHS["vectorizer"], "rb") as f:
        vectorizer = pickle.load(f)
    index = faiss.read_index(PATHS["faiss_tfidf"])
    return vectorizer, index


@st.cache_resource(show_spinner="Loading Sentence Transformer model...")
def load_semantic():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(PATHS["faiss_semantic"])
    return model, index


@st.cache_resource(show_spinner="Loading metadata...")
def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(PATHS["metadata"], index_col="faiss_idx")
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
    /* Result card */
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

    /* Card title */
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-bottom: 4px;
    }

    /* Card meta */
    .card-meta {
        font-size: 0.82rem;
        color: #9BA3B2;
        margin-bottom: 8px;
    }

    /* Card preview */
    .card-preview {
        font-size: 0.88rem;
        color: #C5CAD6;
        line-height: 1.55;
    }

    /* Score badge */
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

    /* Source tag */
    .source-tag {
        display: inline-block;
        background: #2A3050;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        color: #9BA3B2;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }

    /* Divider */
    hr { border-color: #2A3050 !important; }

    /* Query box label */
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
    authors  = row.get("authors", "Unknown")
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
    """
    Run query through selected method, apply source filter,
    return top_n results.
    """
    # Clean the query the same way we cleaned training text
    cleaned_query = clean_text(query, remove_stopwords=True, lemmatize=False)
    if not cleaned_query.strip():
        cleaned_query = query   # fallback: use raw query if cleaning kills it

    fetch_n = top_n * 4   # fetch extra to allow for post-filtering

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
            query_text=query,   # semantic model works better on raw text
            model=sem_model,
            index=sem_index,
            metadata=metadata,
            top_n=fetch_n,
        )

    # ── Source filter ────────────────────────────────────────────────────────
    if source_filter == "Books only":
        results = results[results["source"] == "book"]
    elif source_filter == "Papers only":
        results = results[results["source"] == "paper"]

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
        options=["Semantic (Best Quality)", "TF-IDF (Keyword Match)", "Compare Both"],
        index=0,
        help=(
            "**Semantic**: Uses Sentence Transformers to find meaning-based similarity.\n\n"
            "**TF-IDF**: Finds items sharing exact keywords.\n\n"
            "**Compare Both**: Shows results side by side."
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
    "Reinforcement learning agents that learn to play Atari games from raw pixels",
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

    # ── Load models ───────────────────────────────────────────────────────────
    vectorizer, tfidf_index = load_tfidf()
    sem_model,  sem_index   = load_semantic()
    metadata                = load_metadata()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    settings = render_sidebar()

    # Dataset stats in sidebar
    n_books  = (metadata["source"] == "book").sum()
    n_papers = (metadata["source"] == "paper").sum()
    st.sidebar.metric("📖 Books",   f"{n_books:,}")
    st.sidebar.metric("🔬 Papers",  f"{n_papers:,}")
    st.sidebar.metric("📄 Total",   f"{len(metadata):,}")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Model**: `all-MiniLM-L6-v2`\n\n"
        "**Index**: FAISS IndexFlatIP\n\n"
        "**Dim**: 384 (semantic) | 10k (TF-IDF)"
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
        search_clicked = st.button("🔍 Find Similar", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=False):
            st.rerun()

    st.markdown("---")

    # ── Run recommendations ───────────────────────────────────────────────────
    if search_clicked and query.strip():

        method        = settings["method"]
        source_filter = settings["source_filter"]
        top_n         = settings["top_n"]

        # ── Single method ────────────────────────────────────────────────────
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
                    f"<span style='color:{color};font-size:0.85rem;'>"
                    f"({len(results)} recommendations)</span>",
                    unsafe_allow_html=True,
                )
                for rank, (_, row) in enumerate(results.iterrows(), start=1):
                    render_card(row, rank, method_key)

        # ── Compare Both ─────────────────────────────────────────────────────
        else:
            with st.spinner("Running both methods for comparison..."):
                tfidf_results = get_recommendations(
                    query=query,
                    method="TF-IDF",
                    source_filter=source_filter,
                    top_n=top_n,
                    vectorizer=vectorizer,
                    tfidf_index=tfidf_index,
                    sem_model=sem_model,
                    sem_index=sem_index,
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
                st.markdown(
                    f"### 🔵 TF-IDF Results",
                    unsafe_allow_html=True,
                )
                if tfidf_results.empty:
                    st.warning("No TF-IDF results found.")
                else:
                    for rank, (_, row) in enumerate(tfidf_results.iterrows(), start=1):
                        render_card(row, rank, "tfidf")

            with col_sem:
                st.markdown(
                    f"### 🟠 Semantic Results",
                    unsafe_allow_html=True,
                )
                if sem_results.empty:
                    st.warning("No semantic results found.")
                else:
                    for rank, (_, row) in enumerate(sem_results.iterrows(), start=1):
                        render_card(row, rank, "semantic")

    elif search_clicked and not query.strip():
        st.warning("⚠️ Please enter a query before searching.")

    else:
        # Landing state — show instructions
        st.markdown("""
        <div style="text-align:center; padding: 3rem 0; color: #9BA3B2;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
            <h3 style="color:#C5CAD6;">How to use this app</h3>
            <p>1. Type a book description or paper abstract in the query box above</p>
            <p>2. Or pick one of the example queries on the right</p>
            <p>3. Choose a recommendation method and filters in the sidebar</p>
            <p>4. Click <strong>Find Similar</strong> to get recommendations</p>
            <br>
            <p style="font-size:0.9rem;">
                📖 <strong>Books</strong>: CMU Book Summaries (16k books) &nbsp;|&nbsp;
                🔬 <strong>Papers</strong>: arXiv CS/ML/AI papers
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()