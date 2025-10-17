"""Environment variable helpers shared across preprocessing scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional

_LOADED_PATHS: set[Path] = set()


def locate_env_file(start: Optional[Path] = None, explicit: Optional[Path] = None) -> Optional[Path]:
    """Return the path to a .env file by walking parents, or an explicit file if provided."""
    if explicit:
        candidate = explicit.expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f".env file not found: {candidate}")

    cursor = (start or Path.cwd()).resolve()
    for parent in [cursor, *cursor.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None


def _parse_env(text: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def load_env(explicit: Optional[Path] = None, *, start: Optional[Path] = None, override: bool = False) -> Optional[Path]:
    """
    Load key/value pairs from a .env file into os.environ.

    By default the first .env discovered by walking up from ``start`` (or CWD) is used.
    Existing environment variables are preserved unless ``override`` is True.
    """
    path = locate_env_file(start=start, explicit=explicit)
    if not path:
        return None
    if path in _LOADED_PATHS and not override:
        return path

    data = _parse_env(path.read_text(encoding="utf-8"))
    for key, value in data.items():
        if not override and key in os.environ:
            continue
        os.environ[key] = value
    _LOADED_PATHS.add(path)
    return path


def require_env(names: Iterable[str]) -> None:
    """Ensure the given environment variables are present, raising if any are missing."""
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


__all__ = ["load_env", "require_env", "locate_env_file"]

