"""Shared path helpers for the preprocessing pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

_UTILS_DIR = Path(__file__).resolve().parent
PREPROCESS_ROOT = _UTILS_DIR.parent
REPO_ROOT = PREPROCESS_ROOT.parent

DATA_DIR = PREPROCESS_ROOT / "data"
OUTPUT_DIR = PREPROCESS_ROOT / "output"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
LOG_DIR = PREPROCESS_ROOT / "logs"
CACHE_DIR = PREPROCESS_ROOT / ".cache"


def _default_app_data_dir() -> Path:
    """
    Determine a per-user application data directory where mutable artifacts
    (e.g. SQLite databases) can live without polluting the repository.
    """
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        if base:
            return Path(base) / "WISSpectrrum"
        return Path.home() / "AppData" / "Local" / "WISSpectrrum"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "WISSpectrrum"
    # Linux and other POSIX platforms
    xdg_data_home = os.getenv("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "WISSpectrrum"
    return Path.home() / ".local" / "share" / "WISSpectrrum"


def _resolve_state_db_path() -> Path:
    """
    Resolve the default pipeline job database location.

    Users can override the location via the ``WISS_PIPELINE_DB`` environment
    variable. Otherwise we fall back to a per-user application data directory.
    """
    override = os.getenv("WISS_PIPELINE_DB")
    if override:
        return Path(override).expanduser().resolve()
    return _default_app_data_dir() / "pipeline_state.db"


def resolve_data_roots() -> List[Path]:
    """
    Determine candidate base directories that hold input PDFs.

    The ``WISS_DATA_ROOT`` environment variable (comma-separated) takes
    precedence. When unset we fall back to the repository's ``thesis`` directory.
    """
    env_value = os.getenv("WISS_DATA_ROOT", "").strip()
    roots: List[Path] = []
    if env_value:
        for raw in env_value.split(","):
            if raw.strip():
                roots.append(Path(raw.strip()).expanduser().resolve())
    if not roots:
        default = REPO_ROOT / "thesis"
        roots.append(default.resolve())
    return roots


STATE_DB_PATH = _resolve_state_db_path()
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
    "resolve_data_roots",
    "ensure_preprocess_path",
]
