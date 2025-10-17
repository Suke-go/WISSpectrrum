#!/usr/bin/env python3
"""
End-to-end orchestration for the PDF summarisation pipeline.

This script discovers PDF files, enqueues them in an on-disk job store, and runs
`summarise_pdf.summarise_pdf` to produce summaries, embeddings, and ACM CCS
classifications in a single pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from utils.env import load_env  # noqa: E402
from utils.paths import (  # noqa: E402
    ACM_TAXONOMY_PATH,
    STATE_DB_PATH,
    SUMMARIES_DIR,
    ensure_preprocess_path,
)
from utils.state import (  # noqa: E402
    PipelineState,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
)

ensure_preprocess_path()
from summary.summarize_pdf import summarise_pdf  # type: ignore  # noqa: E402
DEFAULT_DB_PATH = STATE_DB_PATH
DEFAULT_OUTPUT_DIR = SUMMARIES_DIR
DEFAULT_TAXONOMY_PATH = ACM_TAXONOMY_PATH


@dataclass
class SummariserConfig:
    extractor: str = "pypdf"
    grobid_url: Optional[str] = None
    grobid_timeout: float = 5.0
    model: str = "gpt-5-mini"
    language: str = "Japanese"
    chunk_size: int = 2500
    overlap: int = 250
    temperature: float = 0.2
    chunk_max_tokens: int = 1600
    final_max_tokens: int = 1200
    metadata_chars: int = 4000
    dual_language: bool = True
    compute_embeddings: bool = True
    section_embeddings: bool = False
    embedding_provider: str = "gemini"
    embedding_model: Optional[str] = None
    embedding_normalize: bool = True
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None
    vertex_embedding_model: str = "text-embedding-004"
    vertex_embedding_dim: Optional[int] = None
    gemini_api_key: Optional[str] = None
    gemini_embedding_model: str = "models/text-embedding-004"
    gemini_task_type: str = "SEMANTIC_SIMILARITY"
    gemini_batch_size: int = 32
    classify_ccs: bool = True
    ccs_taxonomy_path: Path = field(default_factory=lambda: DEFAULT_TAXONOMY_PATH)
    ccs_model: str = "gpt-5"
    ccs_max_concepts: int = 3
    ccs_top_candidates: int = 15
    ccs_fallback_candidates: int = 25
    ccs_temperature: float = 0.1
    ccs_max_output_tokens: int = 900
    ccs_embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"

    def apply_overrides(self, overrides: Dict[str, object]) -> None:
        for key, value in overrides.items():
            if not hasattr(self, key):
                continue
            if key.endswith("_path") and value not in (None, ""):
                value = Path(value)
            setattr(self, key, value)


def load_config_file(path: Optional[Path]) -> Dict[str, object]:
    if not path:
        return {}
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    if resolved.suffix.lower() not in {".json"}:
        raise ValueError("Only JSON config files are supported at the moment.")
    content = resolved.read_text(encoding="utf-8")
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object at the top level.")
    return data


def discover_pdfs(pdf_args: Sequence[Path], dir_args: Sequence[Path], pattern: str) -> List[Path]:
    discovered: List[Path] = []
    seen: Set[Path] = set()
    for candidate in pdf_args:
        path = candidate.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file: {path}")
        if path not in seen:
            discovered.append(path)
            seen.add(path)

    for directory in dir_args:
        root = directory.expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Directory not found: {root}")
        for path in sorted(root.rglob(pattern)):
            if path.is_file() and path.suffix.lower() == ".pdf":
                resolved = path.resolve()
                if resolved not in seen:
                    discovered.append(resolved)
                    seen.add(resolved)
    return discovered


def choose_summary_path(pdf_path: Path, *, output_dir: Path, bases: Sequence[Path]) -> Path:
    for base in bases:
        try:
            relative = pdf_path.relative_to(base)
            break
        except ValueError:
            continue
    else:
        relative = Path(pdf_path.name)
    return output_dir / relative.with_suffix(".json")


def enqueue_jobs(state: PipelineState, pdfs: Sequence[Path], *, output_dir: Path, base_dirs: Sequence[Path], force: bool) -> Tuple[int, int]:
    enqueued = 0
    skipped = 0
    stats = {STATUS_DONE, STATUS_RUNNING}
    for pdf in pdfs:
        summary_path = choose_summary_path(pdf, output_dir=output_dir, bases=base_dirs)
        metadata = {
            "source_size": pdf.stat().st_size,
            "source_mtime": pdf.stat().st_mtime,
        }
        job_id = state.enqueue(pdf, summary_path, metadata=metadata, force=force)
        job = state.fetch(job_id)
        if job and job.status in stats and not force:
            skipped += 1
        else:
            enqueued += 1
    return enqueued, skipped


def run_jobs(
    state: PipelineState,
    config: SummariserConfig,
    *,
    pretty: bool,
    limit: Optional[int] = None,
) -> Tuple[int, int]:
    processed = 0
    failed = 0
    while True:
        if limit is not None and processed >= limit:
            break
        job = state.next_queued()
        if job is None:
            break
        state.mark_running(job.job_id)
        try:
            embedding_model = config.embedding_model
            if not embedding_model:
                if config.embedding_provider == "gemini":
                    embedding_model = config.gemini_embedding_model
                elif config.embedding_provider == "local":
                    embedding_model = "intfloat/multilingual-e5-large-instruct"
                else:
                    embedding_model = config.vertex_embedding_model
            record = summarise_pdf(
                job.pdf_path,
                extractor=config.extractor,
                grobid_url=config.grobid_url,
                grobid_timeout=config.grobid_timeout,
                model=config.model,
                language=config.language,
                chunk_size=config.chunk_size,
                overlap=config.overlap,
                temperature=config.temperature,
                chunk_max_tokens=config.chunk_max_tokens,
                final_max_tokens=config.final_max_tokens,
                metadata_chars=config.metadata_chars,
                paper_id=None,
                title=None,
                authors=None,
                year=None,
                pdf_link=None,
                code_link=None,
                ccs_paths_cli=None,
                ccs_ids_cli=None,
                compute_embeddings=config.compute_embeddings,
                section_embeddings=config.section_embeddings,
                embedding_model=embedding_model,
                embedding_normalize=config.embedding_normalize,
                embedding_provider=config.embedding_provider,
                vertex_project=config.vertex_project,
                vertex_location=config.vertex_location,
                vertex_embedding_model=config.vertex_embedding_model,
                vertex_embedding_dim=config.vertex_embedding_dim,
                gemini_api_key=config.gemini_api_key,
                gemini_task_type=config.gemini_task_type,
                gemini_batch_size=config.gemini_batch_size,
                classify_ccs=config.classify_ccs,
                ccs_taxonomy_path=config.ccs_taxonomy_path,
                ccs_model=config.ccs_model,
                ccs_max_concepts=config.ccs_max_concepts,
                ccs_top_candidates=config.ccs_top_candidates,
                ccs_fallback_candidates=config.ccs_fallback_candidates,
                ccs_temperature=config.ccs_temperature,
                ccs_max_output_tokens=config.ccs_max_output_tokens,
                ccs_embedding_model=config.ccs_embedding_model,
                dual_language=config.dual_language,
            )
            json_text = json.dumps(record, ensure_ascii=False, indent=2 if pretty else None)
            job.summary_path.parent.mkdir(parents=True, exist_ok=True)
            job.summary_path.write_text(json_text + ("\n" if pretty else ""), encoding="utf-8")
            state.mark_done(job.job_id)
            processed += 1
            print(f"[OK] {job.pdf_path} -> {job.summary_path}")
        except Exception as exc:
            state.mark_failed(job.job_id, str(exc))
            failed += 1
            print(f"[ERROR] {job.pdf_path}: {exc}", file=sys.stderr)
    return processed, failed


def handle_run(args: argparse.Namespace) -> int:
    load_env(explicit=args.env_file)
    pdfs = discover_pdfs(args.pdf or [], args.pdf_dir or [], args.pattern)
    if not pdfs:
        print("[WARN] No PDF files discovered.")
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = args.db.expanduser().resolve()
    state = PipelineState(db_path)
    try:
        base_dirs = [path.expanduser().resolve() for path in (args.pdf_dir or []) if path]
        enqueued, skipped = enqueue_jobs(state, pdfs, output_dir=output_dir, base_dirs=base_dirs, force=args.force)
        print(f"[INFO] Enqueued {enqueued} job(s); skipped {skipped} existing job(s).")

        config = SummariserConfig()
        config_data = load_config_file(args.config)
        if config_data:
            config.apply_overrides(config_data)

        overrides: Dict[str, object] = {}
        if args.model:
            overrides["model"] = args.model
        if args.language:
            overrides["language"] = args.language
        if args.embedding_provider:
            overrides["embedding_provider"] = args.embedding_provider
        if args.disable_embeddings:
            overrides["compute_embeddings"] = False
        if args.disable_ccs:
            overrides["classify_ccs"] = False
        if args.dual_language:
            overrides["dual_language"] = True
        if args.chunk_size is not None:
            overrides["chunk_size"] = args.chunk_size
        if args.temperature is not None:
            overrides["temperature"] = args.temperature
        if overrides:
            config.apply_overrides(overrides)

        processed, failed = run_jobs(
            state,
            config,
            pretty=args.pretty,
            limit=args.limit,
        )
        print(f"[INFO] Completed {processed} job(s); {failed} failed.")
        return 0 if failed == 0 else 2
    finally:
        state.close()


def handle_status(args: argparse.Namespace) -> int:
    db_path = args.db.expanduser().resolve()
    state = PipelineState(db_path)
    try:
        totals = {status: 0 for status in (STATUS_QUEUED, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED)}
        for job in state.list_jobs():
            totals[job.status] = totals.get(job.status, 0) + 1
        print(json.dumps(totals, ensure_ascii=False, indent=2))
    finally:
        state.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the PDF summarisation pipeline end-to-end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Discover PDFs, enqueue jobs, and process the queue.")
    run_parser.add_argument("--pdf", type=Path, action="append", help="Explicit PDF file to process (repeatable).")
    run_parser.add_argument("--pdf-dir", type=Path, action="append", help="Directory to search for PDFs (repeatable).")
    run_parser.add_argument("--pattern", default="*.pdf", help="Glob pattern when scanning directories.")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for summary JSON output.")
    run_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite database used for job tracking.")
    run_parser.add_argument("--config", type=Path, help="Optional JSON file overriding summariser settings.")
    run_parser.add_argument("--env-file", type=Path, help="Optional .env file containing API keys.")
    run_parser.add_argument("--force", action="store_true", help="Requeue jobs even if previously completed.")
    run_parser.add_argument("--limit", type=int, help="Process at most this many queued jobs.")
    run_parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    run_parser.add_argument("--model", help="Override the OpenAI model used for summarisation.")
    run_parser.add_argument("--language", help="Override summary language.")
    run_parser.add_argument("--dual-language", action="store_true", help="Generate English translations in addition to the primary language.")
    run_parser.add_argument("--embedding-provider", choices=["local", "vertex-ai", "gemini"], help="Override embedding provider.")
    run_parser.add_argument("--disable-embeddings", action="store_true", help="Skip embedding generation.")
    run_parser.add_argument("--disable-ccs", action="store_true", help="Skip ACM CCS classification.")
    run_parser.add_argument("--chunk-size", type=int, help="Override chunk size used for summarisation.")
    run_parser.add_argument(
        "--temperature",
        type=float,
        help="Override sampling temperature (ignored by GPT-5 family models).",
    )
    run_parser.set_defaults(func=handle_run)

    status_parser = subparsers.add_parser("status", help="Print counts of jobs by status.")
    status_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite database used for job tracking.")
    status_parser.set_defaults(func=handle_status)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
