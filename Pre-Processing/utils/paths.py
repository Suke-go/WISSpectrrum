"""Shared path helpers for the preprocessing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_UTILS_DIR = Path(__file__).resolve().parent
PREPROCESS_ROOT = _UTILS_DIR.parent
REPO_ROOT = PREPROCESS_ROOT.parent

DATA_DIR = PREPROCESS_ROOT / "data"
OUTPUT_DIR = PREPROCESS_ROOT / "output"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
LOG_DIR = PREPROCESS_ROOT / "logs"
CACHE_DIR = PREPROCESS_ROOT / ".cache"

STATE_DB_PATH = DATA_DIR / "pipeline_state.db"
ACM_TAXONOMY_PATH = PREPROCESS_ROOT / "ACM CCS" / "acm_ccs2012-1626988337597.xml"


def ensure_preprocess_path(path: Optional[Path] = None) -> None:
    """
    Make sure the preprocessing root is present in ``sys.path``.

    This is primarily useful for entrypoints that live inside nested directories
    (e.g. ``summary``) so they can import sibling modules such as ``embeddings``.
    """
    candidate = str((path or PREPROCESS_ROOT).resolve())
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


__all__ = [
    "PREPROCESS_ROOT",
    "REPO_ROOT",
    "DATA_DIR",
    "OUTPUT_DIR",
    "SUMMARIES_DIR",
    "LOG_DIR",
    "CACHE_DIR",
    "STATE_DB_PATH",
    "ACM_TAXONOMY_PATH",
    "ensure_preprocess_path",
]
