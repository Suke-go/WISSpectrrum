#!/usr/bin/env python3
"""
Batch runner that applies summarize_pdf.py to every PDF inside a directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

THIS_DIR = Path(__file__).resolve().parent
PREPROCESS_ROOT = THIS_DIR.parent
if str(PREPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_ROOT))

try:
    from summary.summarize_pdf import parse_args as summary_parse_args  # type: ignore[attr-defined]  # noqa: E402
    from summary.summarize_pdf import summarise_pdf  # type: ignore[attr-defined]  # noqa: E402
except ImportError:
    from summarize_pdf import parse_args as summary_parse_args  # type: ignore[attr-defined]  # noqa: E402
    from summarize_pdf import summarise_pdf  # type: ignore[attr-defined]  # noqa: E402
from utils.env import load_env  # noqa: E402

SUMMARY_DEFAULTS = summary_parse_args(["dummy.pdf"])


def collect_pdfs(base_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    candidates: Iterable[Path]
    if recursive:
        candidates = base_dir.rglob(pattern)
    else:
        candidates = base_dir.glob(pattern)
    return sorted(path for path in candidates if path.is_file())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarise every PDF in a directory using the summarize_pdf pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path, help="Directory that contains PDF files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where JSON summaries will be written (mirrors input structure).",
    )
    parser.add_argument("--pattern", default="*.pdf", help="Filename glob used to select PDFs.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when locating PDFs.")
    parser.add_argument("--env-file", type=Path, help="Optional .env file that provides OPENAI_API_KEY.")
    parser.add_argument(
        "--extractor",
        choices=["pypdf", "grobid"],
        default="grobid",
        help="PDF extraction backend for every file (default: grobid).",
    )
    parser.add_argument("--grobid-url", default=SUMMARY_DEFAULTS.grobid_url, help="Base URL for the GROBID service.")
    parser.add_argument(
        "--grobid-timeout",
        type=float,
        default=SUMMARY_DEFAULTS.grobid_timeout,
        help="Seconds to wait for GROBID health checks and requests.",
    )
    parser.add_argument("--model", default=SUMMARY_DEFAULTS.model, help="OpenAI Responses model.")
    parser.add_argument("--language", default=SUMMARY_DEFAULTS.language, help="Language for generated summaries.")
    parser.add_argument("--chunk-size", type=int, default=SUMMARY_DEFAULTS.chunk_size, help="Characters per chunk.")
    parser.add_argument("--overlap", type=int, default=SUMMARY_DEFAULTS.overlap, help="Overlap between chunks.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=SUMMARY_DEFAULTS.temperature,
        help="Sampling temperature for all LLM calls.",
    )
    parser.add_argument(
        "--chunk-max-output",
        type=int,
        default=SUMMARY_DEFAULTS.chunk_max_output,
        help="Token budget for each chunk summary request.",
    )
    parser.add_argument(
        "--final-max-output",
        type=int,
        default=SUMMARY_DEFAULTS.final_max_output,
        help="Token budget for the final synthesis request.",
    )
    parser.add_argument(
        "--metadata-chars",
        type=int,
        default=SUMMARY_DEFAULTS.metadata_chars,
        help="Characters from the PDF start used for metadata extraction.",
    )
    parser.add_argument("--dual-language", action="store_true", help="Add English translations to the JSON.")
    parser.add_argument(
        "--flatten-translations",
        action="store_true",
        help="Copy English translations into top-level *_en fields.",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Compute embeddings for each summary section.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["local", "vertex-ai", "gemini"],
        default=SUMMARY_DEFAULTS.embedding_provider,
        help="Embedding backend to use when --embeddings is set.",
    )
    parser.add_argument(
        "--embedding-model",
        default=SUMMARY_DEFAULTS.embedding_model,
        help="SentenceTransformer model used for local embeddings.",
    )
    parser.add_argument(
        "--no-embedding-normalize",
        dest="embedding_normalize",
        action="store_false",
        help="Disable L2 normalisation of embeddings.",
    )
    parser.set_defaults(embedding_normalize=SUMMARY_DEFAULTS.embedding_normalize)
    parser.add_argument("--vertex-project", default=SUMMARY_DEFAULTS.vertex_project, help="Vertex AI project ID.")
    parser.add_argument("--vertex-location", default=SUMMARY_DEFAULTS.vertex_location, help="Vertex AI region.")
    parser.add_argument(
        "--vertex-embedding-model",
        default=SUMMARY_DEFAULTS.vertex_embedding_model,
        help="Vertex AI embedding model name.",
    )
    parser.add_argument(
        "--vertex-embedding-dim",
        type=int,
        default=SUMMARY_DEFAULTS.vertex_embedding_dim,
        help="Optional output dimensionality for Vertex embeddings.",
    )
    parser.add_argument("--gemini-api-key", default=SUMMARY_DEFAULTS.gemini_api_key, help="Explicit Gemini API key.")
    parser.add_argument(
        "--gemini-embedding-model",
        default=SUMMARY_DEFAULTS.gemini_embedding_model,
        help="Gemini embedding model identifier.",
    )
    parser.add_argument(
        "--gemini-task-type",
        default=SUMMARY_DEFAULTS.gemini_task_type,
        help="Gemini embedding task type.",
    )
    parser.add_argument(
        "--gemini-batch-size",
        type=int,
        default=SUMMARY_DEFAULTS.gemini_batch_size,
        help="Maximum sections per Gemini embeddings request.",
    )
    parser.add_argument("--classify-ccs", action="store_true", help="Enable ACM CCS classification.")
    parser.add_argument(
        "--ccs-taxonomy",
        type=Path,
        default=SUMMARY_DEFAULTS.ccs_taxonomy,
        help="Path to the ACM CCS taxonomy XML file.",
    )
    parser.add_argument("--ccs-model", default=SUMMARY_DEFAULTS.ccs_model, help="OpenAI model for CCS classification.")
    parser.add_argument(
        "--ccs-max-concepts",
        type=int,
        default=SUMMARY_DEFAULTS.ccs_max_concepts,
        help="Maximum CCS concepts to assign.",
    )
    parser.add_argument(
        "--ccs-top-candidates",
        type=int,
        default=SUMMARY_DEFAULTS.ccs_top_candidates,
        help="Candidate concepts to surface to the LLM.",
    )
    parser.add_argument(
        "--ccs-fallback-candidates",
        type=int,
        default=SUMMARY_DEFAULTS.ccs_fallback_candidates,
        help="Candidate concepts to surface when embeddings are unavailable.",
    )
    parser.add_argument(
        "--ccs-temperature",
        type=float,
        default=SUMMARY_DEFAULTS.ccs_temperature,
        help="Sampling temperature for CCS classification.",
    )
    parser.add_argument(
        "--ccs-max-output",
        type=int,
        default=SUMMARY_DEFAULTS.ccs_max_output,
        help="Token budget for CCS classification responses.",
    )
    parser.add_argument(
        "--ccs-embedding-model",
        default=SUMMARY_DEFAULTS.ccs_embedding_model,
        help="SentenceTransformer model for CCS candidate retrieval.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip PDFs that already have a JSON file.")
    parser.add_argument("--limit", type=int, help="Maximum number of PDFs to process.")
    parser.add_argument("--dry-run", action="store_true", help="List the PDFs without running the summariser.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort on the first failure.")
    parser.add_argument("--verbose", action="store_true", help="Print progress messages.")
    return parser


def build_summary_kwargs(args: argparse.Namespace) -> dict:
    ccs_taxonomy_path = args.ccs_taxonomy
    if isinstance(ccs_taxonomy_path, Path):
        ccs_taxonomy_path = ccs_taxonomy_path.expanduser().resolve()
    return {
        "extractor": args.extractor,
        "grobid_url": args.grobid_url,
        "grobid_timeout": args.grobid_timeout,
        "model": args.model,
        "language": args.language,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "temperature": args.temperature,
        "chunk_max_tokens": args.chunk_max_output,
        "final_max_tokens": args.final_max_output,
        "metadata_chars": args.metadata_chars,
        "paper_id": None,
        "title": None,
        "authors": None,
        "year": None,
        "pdf_link": None,
        "code_link": None,
        "ccs_paths_cli": None,
        "ccs_ids_cli": None,
        "compute_embeddings": args.embeddings,
        "embedding_model": args.embedding_model,
        "embedding_normalize": args.embedding_normalize,
        "embedding_provider": args.embedding_provider,
        "vertex_project": args.vertex_project,
        "vertex_location": args.vertex_location,
        "vertex_embedding_model": args.vertex_embedding_model,
        "vertex_embedding_dim": args.vertex_embedding_dim,
        "gemini_api_key": args.gemini_api_key,
        "gemini_embedding_model": args.gemini_embedding_model,
        "gemini_task_type": args.gemini_task_type,
        "gemini_batch_size": args.gemini_batch_size,
        "classify_ccs": args.classify_ccs,
        "ccs_taxonomy_path": ccs_taxonomy_path,
        "ccs_model": args.ccs_model,
        "ccs_max_concepts": args.ccs_max_concepts,
        "ccs_top_candidates": args.ccs_top_candidates,
        "ccs_fallback_candidates": args.ccs_fallback_candidates,
        "ccs_temperature": args.ccs_temperature,
        "ccs_max_output_tokens": args.ccs_max_output,
        "ccs_embedding_model": args.ccs_embedding_model,
        "dual_language": args.dual_language,
        "flatten_translations": args.flatten_translations,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    env_path = None
    try:
        env_path = load_env(explicit=args.env_file, start=input_dir)
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}", file=sys.stderr)
    if env_path:
        print(f"[INFO] Loaded environment variables from {env_path}", file=sys.stderr)

    pdfs = collect_pdfs(input_dir, args.pattern, args.recursive)
    if args.limit is not None:
        pdfs = pdfs[: max(0, args.limit)]

    if not pdfs:
        print("[WARN] No PDF files found to process.", file=sys.stderr)
        return 0

    summary_kwargs = build_summary_kwargs(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    processed = 0
    for index, pdf_path in enumerate(pdfs, start=1):
        try:
            relative = pdf_path.relative_to(input_dir)
        except ValueError:
            relative = pdf_path.name

        output_path = (output_dir / relative).with_suffix(".json")
        if args.skip_existing and output_path.exists():
            if args.verbose:
                print(f"[SKIP] {relative} (JSON already exists)", file=sys.stderr)
            continue
        if args.dry_run:
            print(f"[DRY RUN] {relative}", file=sys.stderr)
            continue

        if args.verbose:
            print(f"[INFO] ({index}/{len(pdfs)}) Summarising {relative}", file=sys.stderr)

        try:
            record = summarise_pdf(pdf_path, **summary_kwargs)
        except Exception as exc:
            failures += 1
            print(f"[ERROR] Failed to summarise {relative}: {exc}", file=sys.stderr)
            if args.fail_fast:
                return 1
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        json_text = json.dumps(record, ensure_ascii=False, indent=2 if args.pretty else None)
        output_path.write_text(json_text, encoding="utf-8")
        processed += 1

        if args.verbose:
            print(f"[INFO] Wrote {output_path}", file=sys.stderr)

    if failures:
        print(f"[WARN] Completed with {failures} failures out of {len(pdfs)} PDFs.", file=sys.stderr)
    else:
        print(f"[INFO] Successfully processed {processed} PDF(s).", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
