"""
Unit tests for src/data/ingestion.py
Tests public functions using small temp files — no large datasets needed.
"""
import os
import pandas as pd
import pytest
import tempfile

from src.data.ingestion import ingest_books, run_ingestion


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — create tiny fake input files for testing
# ─────────────────────────────────────────────────────────────────────────────

def make_fake_booksummaries(path: str, n: int = 10):
    """
    Write a minimal booksummaries.txt (tab-separated, no header, 7 cols).
    Matches the exact format ingest_books() expects.
    """
    import json
    rows = []
    for i in range(n):
        genres = json.dumps({f"id{i}": "Fiction", f"id{i}b": "Drama"})
        row = "\t".join([
            str(1000 + i),          # wiki_id
            f"/m/freebase{i}",      # freebase_id
            f"Test Book {i}",       # title
            f"Author {i}",          # author
            f"200{i % 10}",         # pub_date
            genres,                 # genres_raw
            f"This is a long plot summary for book number {i}. " * 5,  # summary
        ])
        rows.append(row)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))


# ─────────────────────────────────────────────────────────────────────────────
# TESTS FOR ingest_books()
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestBooks:

    def test_returns_dataframe(self, tmp_path):
        """ingest_books returns a pandas DataFrame."""
        input_file = str(tmp_path / "books.txt")
        output_file = str(tmp_path / "books_out.csv")
        make_fake_booksummaries(input_file, n=10)
        result = ingest_books(input_file, output_file, sample_size=10)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_required_columns(self, tmp_path):
        """Output DataFrame has all 6 required schema columns."""
        input_file = str(tmp_path / "books.txt")
        output_file = str(tmp_path / "books_out.csv")
        make_fake_booksummaries(input_file, n=10)
        result = ingest_books(input_file, output_file, sample_size=10)
        for col in ["item_id", "source", "title", "text", "authors", "category"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_source_column_is_book(self, tmp_path):
        """All rows have source == 'book'."""
        input_file = str(tmp_path / "books.txt")
        output_file = str(tmp_path / "books_out.csv")
        make_fake_booksummaries(input_file, n=5)
        result = ingest_books(input_file, output_file, sample_size=5)
        assert (result["source"] == "book").all()

    def test_item_id_prefix(self, tmp_path):
        """item_id values start with 'book_'."""
        input_file = str(tmp_path / "books.txt")
        output_file = str(tmp_path / "books_out.csv")
        make_fake_booksummaries(input_file, n=5)
        result = ingest_books(input_file, output_file, sample_size=5)
        assert result["item_id"].str.startswith("book_").all()

    def test_sample_size_respected(self, tmp_path):
        """Returns at most sample_size rows."""
        input_file = str(tmp_path / "books.txt")
        output_file = str(tmp_path / "books_out.csv")
        make_fake_booksummaries(input_file, n=20)
        result = ingest_books(input_file, output_file, sample_size=5)
        assert len(result) <= 5

    def test_output_csv_created(self, tmp_path):
        """CSV file is written to the output path."""
        input_file = str(tmp_path / "books.txt")
        output_file = str(tmp_path / "books_out.csv")
        make_fake_booksummaries(input_file, n=5)
        ingest_books(input_file, output_file, sample_size=5)
        assert os.path.exists(output_file)

    def test_output_csv_readable(self, tmp_path):
        """Written CSV can be read back and has correct shape."""
        input_file = str(tmp_path / "books.txt")
        output_file = str(tmp_path / "books_out.csv")
        make_fake_booksummaries(input_file, n=8)
        ingest_books(input_file, output_file, sample_size=8)
        df_read = pd.read_csv(output_file)
        assert len(df_read) > 0
        assert "title" in df_read.columns

    def test_text_column_non_empty(self, tmp_path):
        """text column (plot summary) has meaningful content."""
        input_file = str(tmp_path / "books.txt")
        output_file = str(tmp_path / "books_out.csv")
        make_fake_booksummaries(input_file, n=5)
        result = ingest_books(input_file, output_file, sample_size=5)
        assert (result["text"].str.len() > 10).all()

    def test_missing_file_raises_error(self, tmp_path):
        """Raises FileNotFoundError if input file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ingest_books(
                raw_path=str(tmp_path / "nonexistent.txt"),
                output_path=str(tmp_path / "out.csv"),
                sample_size=10,
            )

    def test_category_extracted_from_genres(self, tmp_path):
        """category column is populated from genres dict."""
        input_file = str(tmp_path / "books.txt")
        output_file = str(tmp_path / "books_out.csv")
        make_fake_booksummaries(input_file, n=5)
        result = ingest_books(input_file, output_file, sample_size=5)
        assert "category" in result.columns
        # Categories should not be empty strings
        assert result["category"].str.len().gt(0).all()