import os
import ast
import json
import logging
import pandas as pd
from tqdm import tqdm

from src.utils.common import load_params, ensure_dir, validate_file_exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# BOOKS — CMU Book Summaries (booksummaries.txt)
# ─────────────────────────────────────────────────────────────

def ingest_books(raw_path: str, output_path: str, sample_size: int) -> pd.DataFrame:
    """
    Parse the CMU Book Summaries dataset.

    File format: tab-separated, NO header row, 7 columns:
        0: Wikipedia article ID
        1: Freebase ID
        2: Title
        3: Author
        4: Publication date
        5: Genres  (a stringified Python dict, may be empty '{}')
        6: Plot summary
    """
    validate_file_exists(raw_path, "CMU Book Summaries")

    logger.info(f"Parsing books from: {raw_path}")

    columns = [
        "wiki_id", "freebase_id", "title", "author",
        "pub_date", "genres_raw", "summary"
    ]

    df = pd.read_csv(
        raw_path,
        sep="\t",
        header=None,
        names=columns,
        on_bad_lines="skip",   # skip any malformed rows
        encoding="utf-8",
    )

    logger.info(f"Raw books loaded: {len(df)} rows")

    # ── Parse genres from stringified dict ──────────────────────────
    def extract_genres(raw: str) -> str:
        try:
            genre_dict = ast.literal_eval(str(raw))
            genres = list(genre_dict.values())
            return ", ".join(genres[:3]) if genres else "Unknown"
        except Exception:
            return "Unknown"

    df["category"] = df["genres_raw"].apply(extract_genres)

    # ── Drop rows with missing title or summary ──────────────────────
    df = df.dropna(subset=["title", "summary"])
    df = df[df["summary"].str.strip().str.len() > 50]  # keep only meaningful summaries

    # ── Sample ──────────────────────────────────────────────────────
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # ── Build common schema ─────────────────────────────────────────
    df = df.reset_index(drop=True)
    result = pd.DataFrame({
        "item_id":  ["book_" + str(i) for i in range(len(df))],
        "source":   "book",
        "title":    df["title"].str.strip(),
        "text":     df["summary"].str.strip(),
        "authors":  df["author"].fillna("Unknown"),
        "category": df["category"],
    })

    ensure_dir(os.path.dirname(output_path))
    result.to_csv(output_path, index=False)
    logger.info(f"Saved {len(result)} book records → {output_path}")
    return result


# ─────────────────────────────────────────────────────────────
# RESEARCH PAPERS — arXiv Metadata Snapshot
# ─────────────────────────────────────────────────────────────

def ingest_papers(
    raw_path: str,
    output_path: str,
    sample_size: int,
    allowed_categories: list,
) -> pd.DataFrame:
    """
    Stream-parse the arXiv metadata JSON snapshot line by line.

    The file is newline-delimited JSON — each line is one paper (JSON object).
    We NEVER load the whole file into memory (it's ~5 GB).

    Fields used per record:
        id, title, abstract, authors, categories
    """
    validate_file_exists(raw_path, "arXiv Metadata Snapshot")

    allowed_set = set(allowed_categories)
    logger.info(f"Streaming arXiv JSON. Keeping categories: {allowed_set}")
    logger.info("This may take 1–3 minutes — reading 5GB line by line...")

    records = []
    total_seen = 0
    total_kept = 0

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning arXiv papers", unit=" papers"):
            line = line.strip()
            if not line:
                continue

            total_seen += 1

            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            # ── Category filter ─────────────────────────────────────
            paper_cats = paper.get("categories", "")
            paper_cat_list = paper_cats.split() if paper_cats else []
            if not any(c in allowed_set for c in paper_cat_list):
                continue

            # ── Field extraction ────────────────────────────────────
            title    = (paper.get("title", "") or "").replace("\n", " ").strip()
            abstract = (paper.get("abstract", "") or "").replace("\n", " ").strip()
            authors  = paper.get("authors", "Unknown") or "Unknown"
            category = paper_cat_list[0] if paper_cat_list else "unknown"

            # Skip if abstract is too short to be useful
            if len(abstract) < 80:
                continue

            records.append({
                "title":    title,
                "text":     abstract,
                "authors":  authors,
                "category": category,
            })
            total_kept += 1

            # Stop once we have enough
            if total_kept >= sample_size:
                logger.info(f"Reached target of {sample_size} papers — stopping early.")
                break

    logger.info(f"Scanned {total_seen:,} papers. Kept {total_kept:,} matching target categories.")

    if not records:
        raise ValueError(
            "No papers matched the allowed categories. "
            "Check your arxiv_categories list in params.yaml."
        )

    df = pd.DataFrame(records).reset_index(drop=True)
    df.insert(0, "item_id", ["paper_" + str(i) for i in range(len(df))])
    df.insert(1, "source", "paper")

    ensure_dir(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} paper records → {output_path}")
    return df


# ─────────────────────────────────────────────────────────────
# COMBINED INGESTION ENTRY POINT
# ─────────────────────────────────────────────────────────────

def run_ingestion() -> tuple[pd.DataFrame, pd.DataFrame]:
    params = load_params()["ingestion"]

    logger.info("=" * 55)
    logger.info("  STARTING INGESTION PIPELINE")
    logger.info("=" * 55)

    # ── Books ────────────────────────────────────────────────
    logger.info("\n[1/2] BOOKS INGESTION")
    books_df = ingest_books(
        raw_path=params["books_raw_path"],
        output_path=params["books_output_path"],
        sample_size=params["books_sample_size"],
    )

    # ── Papers ───────────────────────────────────────────────
    logger.info("\n[2/2] PAPERS INGESTION")
    papers_df = ingest_papers(
        raw_path=params["arxiv_raw_path"],
        output_path=params["arxiv_output_path"],
        sample_size=params["arxiv_sample_size"],
        allowed_categories=params["arxiv_categories"],
    )

    # ── Summary ──────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("  INGESTION COMPLETE")
    logger.info(f"  Books  : {len(books_df):,} records")
    logger.info(f"  Papers : {len(papers_df):,} records")
    logger.info("=" * 55)

    return books_df, papers_df


if __name__ == "__main__":
    run_ingestion()