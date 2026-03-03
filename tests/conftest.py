"""Pytest fixtures for Advanced RAG tests."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_data_path() -> Path:
    """Return path to data directory."""
    return PROJECT_ROOT / "data"


def get_csv_path() -> Path:
    """Return path to daily_sales.csv."""
    return PROJECT_ROOT / "data" / "structured" / "daily_sales.csv"


def get_text_dir() -> Path:
    """Return path to unstructured product pages directory."""
    return PROJECT_ROOT / "data" / "unstructured"
