"""
Unit tests for src/models/similarity.py
"""
import numpy as np
import pandas as pd
import faiss
import os

from src.models.similarity import (
    build_tfidf,
    build_faiss_index_semantic,
    query_tfidf,
    save_metadata,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

import pytest


@pytest.fixture
def sample_texts():
    return [
        "machine learning neural network deep learning",
        "natural language processing text classification",
        "computer vision image recognition convolutional",
        "reinforcement learning reward policy agent",
        "graph neural network knowledge embedding",
        "transformer attention mechanism bert gpt",
        "fantasy magic wizard dragon kingdom castle",
        "science fiction space alien robot future",
        "mystery detective crime murder investigation",
        "romance love relationship historical novel",
    ]


@pytest.fixture
def sample_metadata():
    return pd.DataFrame({
        "item_id": [f"item_{i}" for i in range(10)],
        "source": ["paper"] * 6 + ["book"] * 4,
        "title": [f"Title {i}" for i in range(10)],
        "authors": [f"Author {i}" for i in range(10)],
        "category": ["cs.LG"] * 6 + ["Fiction"] * 4,
        "text": [f"Original text {i}" for i in range(10)],
    })


@pytest.fixture
def tfidf_components(sample_texts, tmp_path):
    vectorizer, matrix = build_tfidf(
        texts=sample_texts,
        max_features=100,
        ngram_range=(1, 1),
        save_dir=str(tmp_path),
    )
    return vectorizer, matrix


@pytest.fixture
def semantic_index(sample_texts, tmp_path):
    np.random.seed(42)
    dim = 32
    embeddings = np.random.randn(len(sample_texts), dim).astype("float32")
    faiss.normalize_L2(embeddings)
    index = build_faiss_index_semantic(
        embeddings=embeddings,
        save_dir=str(tmp_path),
        nlist=2,
    )
    return index, embeddings, dim


# ── TF-IDF Tests ──────────────────────────────────────────────────────────────

class TestBuildTfidf:

    def test_matrix_shape(self, sample_texts, tfidf_components):
        _, matrix = tfidf_components
        assert matrix.shape[0] == len(sample_texts)

    def test_vectorizer_saved(self, tmp_path, sample_texts):
        build_tfidf(sample_texts, 50, (1, 1), str(tmp_path))
        assert os.path.exists(tmp_path / "tfidf_vectorizer.pkl")
        assert os.path.exists(tmp_path / "tfidf_matrix.npz")

    def test_vectorizer_can_transform_new_text(self, tfidf_components):
        vectorizer, _ = tfidf_components
        result = vectorizer.transform(["new neural network text"])
        assert result.shape[0] == 1


class TestQueryTfidf:

    def test_returns_dataframe(self, tfidf_components, sample_metadata):
        vectorizer, matrix = tfidf_components
        dense = matrix.toarray().astype("float32")
        faiss.normalize_L2(dense)
        index = faiss.IndexFlatIP(dense.shape[1])
        index.add(dense)
        result = query_tfidf(
            query_text="neural network deep learning",
            vectorizer=vectorizer,
            index=index,
            metadata=sample_metadata,
            top_n=3,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 3

    def test_returns_similarity_score_column(self, tfidf_components, sample_metadata):
        vectorizer, matrix = tfidf_components
        dense = matrix.toarray().astype("float32")
        faiss.normalize_L2(dense)
        index = faiss.IndexFlatIP(dense.shape[1])
        index.add(dense)
        result = query_tfidf("machine learning", vectorizer, index, sample_metadata, 3)
        assert "similarity_score" in result.columns
        assert "method" in result.columns
        assert (result["method"] == "tfidf").all()


class TestBuildFaissIndexSemantic:

    def test_index_has_correct_count(self, semantic_index, sample_texts):
        index, _, _ = semantic_index
        assert index.ntotal == len(sample_texts)

    def test_index_file_saved(self, semantic_index, tmp_path):
        assert os.path.exists(tmp_path / "faiss_semantic.index")


class TestSaveMetadata:

    def test_metadata_saved_correctly(self, sample_metadata, tmp_path):
        save_metadata(sample_metadata, str(tmp_path))
        saved = pd.read_csv(tmp_path / "metadata.csv")
        assert "title" in saved.columns
        assert "source" in saved.columns
        assert len(saved) == len(sample_metadata)