#!/usr/bin/env python3
"""CLI shim for the embedding.compute_embeddings entrypoint."""

from __future__ import annotations

from pathlib import Path

from utils.paths import ensure_preprocess_path

ensure_preprocess_path(Path(__file__).resolve().parent)


def main() -> int:
    from embedding.compute_embeddings import main as _main

    return _main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
