import re
import logging
import pandas as pd
import nltk
import spacy

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from src.utils.common import load_params, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Load NLP resources once at module level ──────────────────────────────────
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

try:
    NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    raise OSError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL CLEANING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def remove_html(text: str) -> str:
    """Strip HTML tags using BeautifulSoup."""
    return BeautifulSoup(text, "lxml").get_text(separator=" ")


def remove_latex(text: str) -> str:
    """
    Remove common LaTeX fragments found in arXiv abstracts.
    e.g. $\\mathbb{R}$, \\cite{author2020}, \\textbf{word}
    """
    text = re.sub(r"\$.*?\$", " ", text)              # inline math: $...$
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", " ", text)   # \command{...}
    text = re.sub(r"\\[a-zA-Z]+", " ", text)           # lone \command
    return text


def remove_special_characters(text: str) -> str:
    """Keep only letters, numbers, and spaces."""
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)                   # collapse multiple spaces
    return text.strip()


def remove_stopwords_fn(text: str) -> str:
    """Remove English stopwords."""
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def lemmatize_text(text: str) -> str:
    """Lemmatize using spaCy (e.g. 'running' → 'run', 'wolves' → 'wolf')."""
    doc = NLP(text[:50000])  # spaCy has an internal limit; guard against huge texts
    return " ".join([token.lemma_ for token in doc if not token.is_space])


def truncate_text(text: str, max_len: int) -> str:
    """Truncate to max_len characters, cutting at a word boundary."""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLEANING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(
    text: str,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    max_len: int = 1000,
) -> str:
    """
    Full cleaning pipeline for a single text string.
    Order matters:
      1. Strip HTML           (before lowercasing so tags are recognisable)
      2. Remove LaTeX         (before special-char removal)
      3. Lowercase
      4. Remove special chars
      5. Remove stopwords
      6. Lemmatize
      7. Truncate
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = remove_html(text)
    text = remove_latex(text)
    text = text.lower()
    text = remove_special_characters(text)

    if remove_stopwords:
        text = remove_stopwords_fn(text)

    if lemmatize:
        text = lemmatize_text(text)

    text = truncate_text(text, max_len)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# DATASET-LEVEL OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def merge_datasets(books_path: str, papers_path: str) -> pd.DataFrame:
    """
    Load books_raw.csv and papers_raw.csv, validate their schemas,
    and concatenate into one combined DataFrame.
    """
    logger.info("Loading books and papers CSVs...")
    books  = pd.read_csv(books_path)
    papers = pd.read_csv(papers_path)

    expected_cols = {"item_id", "source", "title", "text", "authors", "category"}
    for name, df in [("books", books), ("papers", papers)]:
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(f"[{name}] Missing required columns: {missing}")

    combined = pd.concat([books, papers], ignore_index=True)
    combined = combined.reset_index(drop=True)

    logger.info(
        f"Merged dataset — Books: {len(books):,} | "
        f"Papers: {len(papers):,} | Total: {len(combined):,}"
    )
    return combined


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str,
    cleaned_col: str,
    remove_stopwords: bool,
    lemmatize: bool,
    min_length: int,
    max_length: int,
) -> pd.DataFrame:
    """
    Apply clean_text() to every row, add a text_clean column,
    and filter out rows that are too short after cleaning.
    """
    logger.info(f"Cleaning {len(df):,} rows (lemmatize={lemmatize}, "
                f"stopwords={remove_stopwords})...")

    # Apply cleaning with a progress log every 1000 rows
    cleaned = []
    for i, text in enumerate(df[text_col].fillna("")):
        cleaned.append(
            clean_text(
                text,
                remove_stopwords=remove_stopwords,
                lemmatize=lemmatize,
                max_len=max_length,
            )
        )
        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i + 1:,} / {len(df):,} rows...")

    df = df.copy()
    df[cleaned_col] = cleaned

    # Drop rows where cleaned text is too short to be useful
    before = len(df)
    df = df[df[cleaned_col].str.len() >= min_length].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with cleaned text < {min_length} chars")

    logger.info(f"Preprocessing complete. Final row count: {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing():
    all_params   = load_params()
    ing_params   = all_params["ingestion"]
    prep_params  = all_params["preprocessing"]

    # ── Step 1: Merge ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 1/3 — MERGING DATASETS")
    logger.info("=" * 55)

    combined = merge_datasets(
        books_path  = ing_params["books_output_path"],
        papers_path = ing_params["arxiv_output_path"],
    )

    ensure_dir("data/processed")
    combined.to_csv(prep_params["combined_raw_path"], index=False)
    logger.info(f"Saved combined_raw.csv → {prep_params['combined_raw_path']}")

    # ── Step 2: Clean ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 2/3 — CLEANING TEXT")
    logger.info("=" * 55)

    processed = preprocess_dataframe(
        df               = combined,
        text_col         = prep_params["text_column"],
        cleaned_col      = prep_params["cleaned_column"],
        remove_stopwords = prep_params["remove_stopwords"],
        lemmatize        = prep_params["lemmatize"],
        min_length       = prep_params["min_text_length"],
        max_length       = prep_params["max_text_length"],
    )

    processed.to_csv(prep_params["combined_processed_path"], index=False)
    logger.info(f"Saved combined_processed.csv → {prep_params['combined_processed_path']}")

    # ── Step 3: Summary ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 3/3 — SUMMARY")
    logger.info("=" * 55)
    logger.info(f"  Total rows         : {len(processed):,}")
    logger.info(f"  Books              : {(processed['source'] == 'book').sum():,}")
    logger.info(f"  Papers             : {(processed['source'] == 'paper').sum():,}")
    logger.info(f"  Columns            : {list(processed.columns)}")
    logger.info(f"  Avg clean text len : {processed['text_clean'].str.len().mean():.0f} chars")
    logger.info("=" * 55)

    return processed


if __name__ == "__main__":
    run_preprocessing()