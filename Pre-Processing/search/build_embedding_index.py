#!/usr/bin/env python3
"""
Collect section-wise embeddings from summary JSON files and write a reusable index.

The summariser emits per-section embeddings (positioning/purpose/method/evaluation/abstract)
when invoked with ``--embeddings``. This script normalises those records into a flat JSONL
structure that downstream tools can load to power semantic search or analytics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


SECTION_FIELD_MAP = {
    "positioning": "positioning_summary",
    "purpose": "purpose_summary",
    "method": "method_summary",
    "evaluation": "evaluation_summary",
    "abstract": "abstract",
}


def resolve_inputs(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_file():
            paths.append(path)
            continue
        if path.is_dir():
            paths.extend(sorted(path.glob("**/*.json")))
            continue
        matches = sorted(Path().glob(raw))
        if matches:
            paths.extend(matches)
            continue
        raise FileNotFoundError(f"No files matched input pattern: {raw}")
    return paths


def iter_records(paths: Iterable[Path], *, min_chars: int) -> Iterable[Dict[str, object]]:
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            print(f"[WARN] Skipping {path}: failed to parse JSON ({exc})")
            continue

        embeddings = data.get("embeddings") or {}
        sections = embeddings.get("sections")
        if not sections:
            continue

        provider = embeddings.get("provider")
        model = embeddings.get("model")
        dim = embeddings.get("dim")
        normed = embeddings.get("normed")

        for section_name, vector in sections.items():
            if not isinstance(vector, list) or not vector:
                continue
            summary_field = SECTION_FIELD_MAP.get(section_name, f"{section_name}_summary")
            text = data.get(summary_field)
            if not isinstance(text, str):
                text = None
            if text:
                text = text.strip()
            if not text or len(text) < min_chars:
                continue

            yield {
                "paper_id": data.get("id"),
                "title": data.get("title"),
                "year": data.get("year"),
                "authors": data.get("authors"),
                "section": section_name,
                "summary_field": summary_field,
                "text": text,
                "embedding": vector,
                "embedding_model": model,
                "embedding_provider": provider,
                "embedding_dim": dim,
                "embedding_normed": normed,
                "source_path": str(path),
            }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect per-section embeddings from summary JSON files into a JSONL index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Summary JSON files, directories, or glob patterns.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("embedding_index.jsonl"),
        help="Where to write the JSONL index.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=40,
        help="Skip sections whose text is shorter than this many characters.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        inputs = resolve_inputs(args.inputs)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not inputs:
        print("[ERROR] No input files found.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for record in iter_records(inputs, min_chars=args.min_chars):
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
            count += 1

    print(f"[INFO] Wrote {count} embeddings to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
