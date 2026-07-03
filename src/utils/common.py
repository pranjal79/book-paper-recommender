import yaml
import os
import logging

logger = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    """Load the params.yaml config file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"params file not found at: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def validate_file_exists(path: str, label: str):
    """Raise a clear error if a required input file is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] {label} not found at: {path}\n"
            f"Please place the file there before running ingestion.\n"
        )
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"Found {label} at '{path}' ({size_mb:.1f} MB)")