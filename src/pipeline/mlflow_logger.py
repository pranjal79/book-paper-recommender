"""
mlflow_logger.py
────────────────
Centralised MLflow tracking helper for the Book & Paper Recommender.

Responsibilities:
  - Connect to DagsHub remote tracking server via environment variables
  - Start / end MLflow runs
  - Log params, metrics, tags, and artifacts in a structured way
  - Provide helper functions that train_pipeline.py calls after each stage
"""

import os
import subprocess
import logging
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import faiss
import mlflow

from dotenv import load_dotenv
from src.utils.common import load_params

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_mlflow() -> str:
    """
    Load DagsHub credentials from .env and configure MLflow tracking URI.
    Returns the experiment name.
    """
    # Load .env (silently — no error if file doesn't exist)
    load_dotenv()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username     = os.getenv("MLFLOW_TRACKING_USERNAME")
    password     = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not all([tracking_uri, username, password]):
        logger.warning(
            "DagsHub credentials not found in .env — "
            "falling back to LOCAL mlflow tracking (mlruns/).\n"
            "Set MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, "
            "MLFLOW_TRACKING_PASSWORD in your .env file to enable remote tracking."
        )
        mlflow.set_tracking_uri("mlruns")
        return "book-paper-recommender-local"

    # Set credentials as env vars (mlflow reads these automatically)
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    mlflow.set_tracking_uri(tracking_uri)

    logger.info(f"MLflow tracking URI set to: {tracking_uri}")
    return "book-paper-recommender"


def get_git_commit() -> str:
    """Return the current short git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# METRIC COMPUTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_data_metrics(params: dict) -> dict:
    """
    Compute metrics from the processed dataset and raw CSVs.
    Called after preprocessing stage.
    """
    metrics = {}
    prep = params["preprocessing"]
    ing  = params["ingestion"]

    try:
        processed = pd.read_csv(prep["combined_processed_path"])
        metrics["total_docs"]       = len(processed)
        metrics["books_count"]      = int((processed["source"] == "book").sum())
        metrics["papers_count"]     = int((processed["source"] == "paper").sum())
        metrics["avg_clean_text_len"] = float(
            processed[prep["cleaned_column"]].str.len().mean()
        )
        metrics["null_text_pct"] = float(
            processed[prep["cleaned_column"]].isnull().mean() * 100
        )

        # Dropped rows
        raw = pd.read_csv(prep["combined_raw_path"])
        metrics["rows_dropped"] = len(raw) - len(processed)

    except Exception as e:
        logger.warning(f"Could not compute data metrics: {e}")

    return metrics


def compute_model_metrics(params: dict) -> dict:
    """
    Compute metrics from the saved model artifacts.
    Called after feature extraction stage.
    """
    metrics = {}
    store = params["features"]["models_store_dir"]

    try:
        # TF-IDF
        vec_path = os.path.join(store, "tfidf_vectorizer.pkl")
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        metrics["tfidf_vocab_size"] = len(vectorizer.vocabulary_)

        mat_path = os.path.join(store, "tfidf_matrix.npz")
        mat = sp.load_npz(mat_path)
        metrics["tfidf_matrix_rows"] = mat.shape[0]
        metrics["tfidf_matrix_cols"] = mat.shape[1]
        metrics["tfidf_sparsity"]    = float(
            1.0 - mat.nnz / (mat.shape[0] * mat.shape[1])
        )

        # Embeddings
        emb_path = os.path.join(store, "sentence_embeddings.npy")
        emb = np.load(emb_path)
        metrics["embedding_dim"]       = emb.shape[1]
        metrics["embedding_count"]     = emb.shape[0]
        metrics["embedding_norm_mean"] = float(
            np.linalg.norm(emb, axis=1).mean()
        )

        # FAISS indices
        tfidf_idx = faiss.read_index(os.path.join(store, "faiss_tfidf.index"))
        sem_idx   = faiss.read_index(os.path.join(store, "faiss_semantic.index"))
        metrics["faiss_tfidf_vectors"]    = tfidf_idx.ntotal
        metrics["faiss_semantic_vectors"] = sem_idx.ntotal

        # Artifact sizes (MB)
        for label, filename in [
            ("size_mb_tfidf_matrix",   "tfidf_matrix.npz"),
            ("size_mb_embeddings",     "sentence_embeddings.npy"),
            ("size_mb_faiss_tfidf",    "faiss_tfidf.index"),
            ("size_mb_faiss_semantic", "faiss_semantic.index"),
            ("size_mb_metadata",       "metadata.csv"),
        ]:
            path = os.path.join(store, filename)
            if os.path.exists(path):
                metrics[label] = round(os.path.getsize(path) / (1024 * 1024), 2)

    except Exception as e:
        logger.warning(f"Could not compute model metrics: {e}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOGGING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def log_pipeline_run(manifest: dict, params: dict):
    """
    Log a complete pipeline run to MLflow / DagsHub.
    Call this at the END of a successful pipeline run.

    Logs:
      - All params from params.yaml
      - Data metrics (row counts, text lengths, etc.)
      - Model metrics (vocab size, embedding dim, FAISS stats)
      - Key artifacts (manifest, processed CSV, model files)
      - Git commit tag
    """
    experiment_name = setup_mlflow()
    mlflow.set_experiment(experiment_name)

    run_name   = params["pipeline"]["run_name"]
    git_commit = get_git_commit()

    logger.info(f"Starting MLflow run: '{run_name}' in experiment: '{experiment_name}'")

    with mlflow.start_run(run_name=run_name) as run:

        # ── Tags ─────────────────────────────────────────────────────────────
        mlflow.set_tags({
            "git_commit":        git_commit,
            "pipeline_version":  "1.0",
            "run_status":        manifest.get("status", "unknown"),
            "developer":         os.getenv("USER", os.getenv("USERNAME", "unknown")),
        })
        logger.info(f"  Tagged run with git_commit={git_commit}")

        # ── Parameters ───────────────────────────────────────────────────────
        _log_params_flat(params)
        logger.info("  Logged parameters from params.yaml")

        # ── Data Metrics ─────────────────────────────────────────────────────
        data_metrics = compute_data_metrics(params)
        if data_metrics:
            mlflow.log_metrics(data_metrics)
            logger.info(f"  Logged data metrics: {list(data_metrics.keys())}")

        # ── Model Metrics ────────────────────────────────────────────────────
        model_metrics = compute_model_metrics(params)
        if model_metrics:
            mlflow.log_metrics(model_metrics)
            logger.info(f"  Logged model metrics: {list(model_metrics.keys())}")

        # ── Artifacts ────────────────────────────────────────────────────────
        _log_artifacts(params)

        run_id = run.info.run_id
        logger.info(f"  ✅ MLflow run completed. Run ID: {run_id}")

    return run_id


def _log_params_flat(params: dict, prefix: str = ""):
    """
    Flatten nested params.yaml dict and log to MLflow.
    MLflow param keys must be strings and values must be < 500 chars.
    Example: params['features']['tfidf_max_features'] → 'features.tfidf_max_features'
    """
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            _log_params_flat(value, prefix=full_key)
        elif isinstance(value, list):
            # Convert lists to comma-separated string
            mlflow.log_param(full_key, ",".join(str(v) for v in value)[:499])
        else:
            mlflow.log_param(full_key, str(value)[:499])


def _log_artifacts(params: dict):
    """Log key output files as MLflow artifacts."""
    store   = params["features"]["models_store_dir"]
    prep    = params["preprocessing"]
    pipeline= params["pipeline"]

    artifacts_to_log = [
        # (local_path, artifact_subdir_in_mlflow)
        (pipeline["manifest_path"],              "pipeline"),
        (prep["combined_processed_path"],         "data"),
        (os.path.join(store, "metadata.csv"),     "models"),
        (os.path.join(store, "tfidf_vectorizer.pkl"), "models"),
        # skip large binary files (embeddings, FAISS) — DVC will handle those
    ]

    for path, artifact_dir in artifacts_to_log:
        if os.path.exists(path):
            mlflow.log_artifact(path, artifact_path=artifact_dir)
            logger.info(f"  Logged artifact: {path} → {artifact_dir}/")
        else:
            logger.warning(f"  Artifact not found, skipping: {path}")