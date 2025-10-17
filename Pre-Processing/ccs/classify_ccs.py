#!/usr/bin/env python3
"""Assign ACM CCS concepts to summarised papers using an LLM-driven workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from classifier import CCSClassifier, summary_to_prompt_text  # noqa: E402
from taxonomy import CCSTaxonomy, load_taxonomy  # noqa: E402
from utils.env import load_env  # type: ignore  # noqa: E402


def load_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit('The "openai" package is required. Install it with "pip install openai>=1.0.0".') from exc
    return OpenAI()


def load_summary(path: Path) -> Dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"Summary file not found: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON at {path}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path, help="Summary JSON files to classify.")
    parser.add_argument("--env-file", type=Path, help="Optional .env file to load before running.")
    parser.add_argument("--xml", type=Path, default=THIS_DIR.parent.parent / "ACM CCS" / "acm_ccs2012-1626988337597.xml", help="Path to the CCS XML file.")
    parser.add_argument(
        "--model",
        default=os.getenv("CCS_CLASSIFIER_MODEL", "gpt-5"),
        help="OpenAI Responses model to use (e.g. 'gpt-5', 'gpt-5-mini').",
    )
    parser.add_argument("--max-concepts", type=int, default=3, help="Maximum number of CCS concepts to assign per paper.")
    parser.add_argument("--top-candidates", type=int, default=15, help="Number of candidate concepts to surface to the LLM.")
    parser.add_argument("--fallback-candidates", type=int, default=25, help="Number of candidates when embeddings are unavailable.")
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("CCS_EMBEDDING_MODEL", "models/text-embedding-004"),
        help="Embedding model identifier (Gemini or SentenceTransformer). Use 'none' to disable embeddings.",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=os.getenv("CCS_EMBEDDING_CACHE"),
        help="Directory to cache CCS concept embeddings (defaults to a .cache folder next to this script).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM decoding temperature (ignored by GPT-5 family models).",
    )
    parser.add_argument("--max-output-tokens", type=int, default=600, help="Maximum tokens for the LLM response.")
    parser.add_argument("--update", action="store_true", help="Update the input summary files in-place with CCS predictions.")
    parser.add_argument("--output", type=Path, help="Optional JSONL report of classifications.")
    return parser


def ensure_classifier(taxonomy: CCSTaxonomy, embedding_model: Optional[str], embedding_cache: Optional[Path]) -> CCSClassifier:
    classifier = CCSClassifier(
        taxonomy,
        embedding_model=embedding_model,
        embedding_cache_dir=embedding_cache,
    )
    return classifier


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    env_path = load_env(explicit=args.env_file, start=Path.cwd())
    if env_path:
        print(f"[INFO] Loaded environment variables from {env_path}", file=sys.stderr)

    taxonomy = load_taxonomy(args.xml)
    embedding_model = None if args.embedding_model == "none" else args.embedding_model
    embedding_cache = Path(args.embedding_cache).expanduser() if args.embedding_cache else None
    classifier = ensure_classifier(taxonomy, embedding_model, embedding_cache)
    client = load_openai_client()

    results: List[Dict[str, object]] = []

    for summary_path in args.summaries:
        record = load_summary(summary_path)
        summary_text = summary_to_prompt_text(record)
        if not summary_text.strip():
            print(f"[WARN] Empty summary context for {summary_path}. Skipping.", file=sys.stderr)
            continue
        try:
            outcome = classifier.classify(
                record,
                client,
                model=args.model,
                max_concepts=args.max_concepts,
                top_candidates=args.top_candidates,
                fallback_candidates=args.fallback_candidates,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                summary_text=summary_text,
            )
        except ValueError as exc:
            print(f"[WARN] {exc} ({summary_path})", file=sys.stderr)
            continue

        results.append(
            {
                "source": str(summary_path),
                "record": outcome.record,
                "predictions": outcome.predictions,
                "candidates": [
                    {
                        "id": candidate.concept.concept_id,
                        "path": candidate.concept.full_path_string(),
                        "score": candidate.score,
                    }
                    for candidate in outcome.candidates
                ],
            }
        )

        if args.update:
            summary_path.write_text(json.dumps(outcome.record, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output and results:
        with args.output.open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    if not results:
        return 1

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
