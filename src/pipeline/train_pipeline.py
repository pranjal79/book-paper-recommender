"""
train_pipeline.py
─────────────────
Master orchestrator for the Book & Paper Recommendation pipeline.

Stages:
  1. Ingestion      → raw CSVs from booksummaries.txt + arXiv JSON
  2. Preprocessing  → combined_raw.csv + combined_processed.csv
  3. Feature Extraction → TF-IDF, embeddings, FAISS indices, metadata

Usage:
  python src/pipeline/train_pipeline.py
  python src/pipeline/train_pipeline.py --skip-ingestion
  python src/pipeline/train_pipeline.py --stages ingestion preprocessing
"""

import os
import sys
import json
import time
import logging
import argparse
import traceback
from datetime import datetime
from pathlib import Path

# ── Make sure project root is on PYTHONPATH when run directly ────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.common import load_params, ensure_dir
from src.data.ingestion import run_ingestion
from src.features.preprocessing import run_preprocessing
from src.models.similarity import run_feature_extraction
from src.pipeline.mlflow_logger import log_pipeline_run

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str, run_name: str) -> logging.Logger:
    ensure_dir(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"{run_name}_{timestamp}.log")

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)s | %(message)s",
        handlers= [
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("pipeline")
    logger.info(f"Logging to: {log_file}")
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# STAGE VALIDATORS
# Validate that each stage produced the expected output files/shapes
# before the next stage starts — fail fast with clear messages.
# ─────────────────────────────────────────────────────────────────────────────

def validate_ingestion(params: dict, logger: logging.Logger) -> dict:
    ing = params["ingestion"]
    results = {}

    for label, path in [
        ("books_raw",  ing["books_output_path"]),
        ("papers_raw", ing["arxiv_output_path"]),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[validate_ingestion] Expected file not found: {path}"
            )
        import pandas as pd
        df = pd.read_csv(path)

        expected_cols = {"item_id", "source", "title", "text", "authors", "category"}
        missing_cols  = expected_cols - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"[validate_ingestion] {label} missing columns: {missing_cols}"
            )
        if len(df) == 0:
            raise ValueError(f"[validate_ingestion] {label} is empty!")

        results[label] = {"rows": len(df), "path": path}
        logger.info(f"  ✅ {label}: {len(df):,} rows at {path}")

    return results


def validate_preprocessing(params: dict, logger: logging.Logger) -> dict:
    prep = params["preprocessing"]
    results = {}

    for label, path in [
        ("combined_raw",       prep["combined_raw_path"]),
        ("combined_processed", prep["combined_processed_path"]),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[validate_preprocessing] Expected file not found: {path}"
            )
        import pandas as pd
        df = pd.read_csv(path)

        if label == "combined_processed":
            if prep["cleaned_column"] not in df.columns:
                raise ValueError(
                    f"[validate_preprocessing] '{prep['cleaned_column']}' column missing"
                )
            null_pct = df[prep["cleaned_column"]].isnull().mean() * 100
            if null_pct > 10:
                logger.warning(
                    f"  ⚠️  {null_pct:.1f}% of cleaned texts are null — check preprocessing"
                )

        results[label] = {"rows": len(df), "path": path}
        logger.info(f"  ✅ {label}: {len(df):,} rows at {path}")

    return results


def validate_feature_extraction(params: dict, logger: logging.Logger) -> dict:
    feat    = params["features"]
    store   = feat["models_store_dir"]
    results = {}

    expected_files = {
        "tfidf_vectorizer": os.path.join(store, "tfidf_vectorizer.pkl"),
        "tfidf_matrix":     os.path.join(store, "tfidf_matrix.npz"),
        "embeddings":       os.path.join(store, "sentence_embeddings.npy"),
        "faiss_tfidf":      os.path.join(store, "faiss_tfidf.index"),
        "faiss_semantic":   os.path.join(store, "faiss_semantic.index"),
        "metadata":         os.path.join(store, "metadata.csv"),
    }

    for label, path in expected_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[validate_features] Expected file not found: {path}"
            )
        size_mb = os.path.getsize(path) / (1024 * 1024)
        results[label] = {"path": path, "size_mb": round(size_mb, 2)}
        logger.info(f"  ✅ {label}: {size_mb:.1f} MB at {path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STAGE RUNNER
# Wraps each stage with timing, error handling, and manifest recording
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(
    stage_name: str,
    stage_fn,
    validate_fn,
    params: dict,
    manifest: dict,
    logger: logging.Logger,
) -> bool:
    """
    Run a single pipeline stage and record its result in the manifest.
    Returns True on success, False on failure.
    """
    banner = f"  STAGE: {stage_name.upper()}  "
    logger.info("\n" + "=" * 60)
    logger.info(banner)
    logger.info("=" * 60)

    stage_start = time.time()
    manifest["stages"][stage_name] = {
        "status":     "running",
        "start_time": datetime.now().isoformat(),
    }

    try:
        # ── Run the stage ────────────────────────────────────────────────────
        stage_fn()

        # ── Validate outputs ─────────────────────────────────────────────────
        logger.info(f"\nValidating {stage_name} outputs...")
        validation_results = validate_fn(params, logger)

        elapsed = time.time() - stage_start
        manifest["stages"][stage_name].update({
            "status":     "success",
            "end_time":   datetime.now().isoformat(),
            "elapsed_s":  round(elapsed, 2),
            "outputs":    validation_results,
        })

        logger.info(
            f"\n✅ Stage '{stage_name}' completed successfully "
            f"in {elapsed:.1f}s"
        )
        return True

    except Exception as e:
        elapsed = time.time() - stage_start
        manifest["stages"][stage_name].update({
            "status":    "failed",
            "end_time":  datetime.now().isoformat(),
            "elapsed_s": round(elapsed, 2),
            "error":     str(e),
            "traceback": traceback.format_exc(),
        })

        logger.error(f"\n❌ Stage '{stage_name}' FAILED after {elapsed:.1f}s")
        logger.error(f"   Error: {e}")
        logger.error(traceback.format_exc())
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MANIFEST HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def init_manifest(params: dict) -> dict:
    return {
        "run_name":   params["pipeline"]["run_name"],
        "start_time": datetime.now().isoformat(),
        "end_time":   None,
        "status":     "running",
        "params":     params,
        "stages":     {},
    }


def save_manifest(manifest: dict, path: str):
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def print_summary(manifest: dict, logger: logging.Logger):
    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE SUMMARY")
    logger.info("=" * 60)
    for stage, info in manifest["stages"].items():
        status  = info.get("status", "unknown")
        elapsed = info.get("elapsed_s", 0)
        icon    = "✅" if status == "success" else ("⏭️" if status == "skipped" else "❌")
        logger.info(f"  {icon}  {stage:<25} {status:<10} {elapsed:.1f}s")
    logger.info(f"\n  Overall : {manifest['status'].upper()}")
    logger.info(f"  Started : {manifest['start_time']}")
    logger.info(f"  Ended   : {manifest['end_time']}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Book & Paper Recommender ML pipeline"
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip the ingestion stage (use already-downloaded data)"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["ingestion", "preprocessing", "feature_extraction"],
        help="Run only specific stages (e.g. --stages preprocessing feature_extraction)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    params = load_params()

    # ── Logging ──────────────────────────────────────────────────────────────
    logger = setup_logging(
        log_dir  = params["pipeline"]["log_dir"],
        run_name = params["pipeline"]["run_name"],
    )

    logger.info("🚀 Starting Book & Paper Recommender Pipeline")
    logger.info(f"   Run name : {params['pipeline']['run_name']}")
    logger.info(f"   Args     : {vars(args)}")

    # ── Manifest ─────────────────────────────────────────────────────────────
    manifest      = init_manifest(params)
    manifest_path = params["pipeline"]["manifest_path"]

    # ── Determine which stages to run ────────────────────────────────────────
    all_stages = ["ingestion", "preprocessing", "feature_extraction"]

    if args.stages:
        stages_to_run = args.stages
    elif args.skip_ingestion:
        stages_to_run = ["preprocessing", "feature_extraction"]
    else:
        stages_to_run = all_stages

    logger.info(f"   Stages to run: {stages_to_run}")

    # ── Stage registry ───────────────────────────────────────────────────────
    stage_registry = {
        "ingestion": {
            "fn":       run_ingestion,
            "validate": validate_ingestion,
        },
        "preprocessing": {
            "fn":       run_preprocessing,
            "validate": validate_preprocessing,
        },
        "feature_extraction": {
            "fn":       run_feature_extraction,
            "validate": validate_feature_extraction,
        },
    }

    # ── Run stages ────────────────────────────────────────────────────────────
    pipeline_success = True

    for stage_name in all_stages:
        if stage_name not in stages_to_run:
            # Mark as skipped in manifest
            manifest["stages"][stage_name] = {
                "status": "skipped",
                "elapsed_s": 0.0,
            }
            logger.info(f"\n⏭️  Skipping stage: {stage_name}")
            continue

        entry   = stage_registry[stage_name]
        success = run_stage(
            stage_name  = stage_name,
            stage_fn    = entry["fn"],
            validate_fn = entry["validate"],
            params      = params,
            manifest    = manifest,
            logger      = logger,
        )

        save_manifest(manifest, manifest_path)  # save after every stage

        if not success:
            pipeline_success = False
            logger.error(
                f"\n🛑 Pipeline aborted at stage '{stage_name}'. "
                f"Fix the error above and re-run."
            )
            break

    # ── Finalize manifest ────────────────────────────────────────────────────
    manifest["status"]   = "success" if pipeline_success else "failed"
    manifest["end_time"] = datetime.now().isoformat()
    save_manifest(manifest, manifest_path)

    # ── Print summary ────────────────────────────────────────────────────────
    print_summary(manifest, logger)

    # ── MLflow Logging ───────────────────────────────────────────────────────
    if pipeline_success:
        logger.info("\n📊 Logging run to MLflow / DagsHub...")
        try:
            run_id = log_pipeline_run(
                manifest=manifest,
                params=params,
            )
            logger.info(f"✅ MLflow run logged successfully. Run ID: {run_id}")
        except Exception as e:
            logger.warning(
                f"⚠️ MLflow logging failed (pipeline still succeeded): {e}\n"
                "Check your .env credentials and DagsHub connectivity."
            )
    else:
        logger.info("⏭️ Skipping MLflow logging — pipeline did not succeed.")
        sys.exit(1)


if __name__ == "__main__":
    main()