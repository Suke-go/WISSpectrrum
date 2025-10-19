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
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from utils.env import load_env  # noqa: E402
from utils.paths import (  # noqa: E402
    ACM_TAXONOMY_PATH,
    STATE_DB_PATH,
    SUMMARIES_DIR,
    ensure_preprocess_path,
    resolve_data_roots,
)
from utils.state import (  # noqa: E402
    PipelineState,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
)

ensure_preprocess_path()
from embeddings import (  # noqa: E402
    EmbeddingQuotaExceeded,
    maybe_compute_embeddings_gemini,
    maybe_compute_embeddings_local,
    maybe_compute_embeddings_vertex_ai,
)
from ccs.classifier import CCSClassifier, summary_to_prompt_text  # noqa: E402
from ccs.taxonomy import load_taxonomy  # noqa: E402
from summary.summarize_pdf import load_openai_client, summarise_pdf  # type: ignore  # noqa: E402
DEFAULT_DB_PATH = STATE_DB_PATH
DEFAULT_OUTPUT_DIR = SUMMARIES_DIR
DEFAULT_TAXONOMY_PATH = ACM_TAXONOMY_PATH


@dataclass
class SummariserConfig:
    extractor: str = "pypdf"
    grobid_url: Optional[str] = None
    grobid_timeout: float = 60.0
    model: str = "gpt-5-mini"
    language: str = "Japanese"
    chunk_size: int = 2500
    overlap: int = 250
    temperature: float = 0.2
    chunk_max_tokens: int = 2200
    final_max_tokens: int = 1600
    metadata_chars: int = 4000
    dual_language: bool = True
    flatten_translations: bool = True
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
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_task_type: str = "SEMANTIC_SIMILARITY"
    gemini_batch_size: int = 32
    classify_ccs: bool = True
    ccs_taxonomy_path: Path = field(default_factory=lambda: DEFAULT_TAXONOMY_PATH)
    ccs_model: str = "gpt-5-mini"
    ccs_max_concepts: int = 3
    ccs_top_candidates: int = 15
    ccs_fallback_candidates: int = 25
    ccs_temperature: float = 0.1
    ccs_max_output_tokens: int = 900
    ccs_embedding_model: Optional[str] = "gemini-embedding-001"

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


def choose_summary_path(
    pdf_path: Path,
    *,
    output_dir: Path,
    bases: Sequence[Path],
) -> Tuple[Path, Optional[Path], Path]:
    for base in bases:
        try:
            relative = pdf_path.relative_to(base)
            return output_dir / relative.with_suffix(".json"), base, relative
        except ValueError:
            continue
    relative = Path(pdf_path.name)
    return output_dir / relative.with_suffix(".json"), None, relative


def _resolve_embedding_model(config: SummariserConfig) -> Optional[str]:
    if config.embedding_model:
        return config.embedding_model
    if config.embedding_provider == "gemini":
        return config.gemini_embedding_model
    if config.embedding_provider == "local":
        return "intfloat/multilingual-e5-large-instruct"
    return config.vertex_embedding_model


def _prepare_embedding_sections(record: Dict[str, object]) -> Dict[str, str]:
    def _usable(value: object) -> Optional[str]:
        if value in (None, "", [], {}):
            return None
        if isinstance(value, list):
            parts = [str(item).strip() for item in value if str(item).strip()]
            return "\n".join(parts) if parts else None
        text = str(value).strip()
        return text or None

    candidates = {
        "overview": record.get("abstract"),
        "positioning": record.get("positioning_summary"),
        "purpose": record.get("purpose_summary"),
        "method": record.get("method_summary"),
        "evaluation": record.get("evaluation_summary"),
        "abstract": record.get("abstract"),
    }
    sections: Dict[str, str] = {}
    for key, candidate in candidates.items():
        normalized = _usable(candidate)
        if normalized:
            sections[key] = normalized
    return sections


def _compute_embeddings_for_record(record: Dict[str, object], config: SummariserConfig) -> Dict[str, object]:
    sections = _prepare_embedding_sections(record)
    if not sections:
        return {}

    provider = config.embedding_provider
    model_name = _resolve_embedding_model(config)

    if provider == "local":
        if not model_name:
            raise ValueError("Embedding model must be specified when using the local provider.")
        return maybe_compute_embeddings_local(
            sections,
            model_name=model_name,
            normalize=config.embedding_normalize,
        )

    if provider == "vertex-ai":
        project = config.vertex_project or os.getenv("VERTEX_AI_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = config.vertex_location or os.getenv("VERTEX_AI_LOCATION") or "us-central1"
        resolved_model = model_name or os.getenv("VERTEX_AI_EMBEDDING_MODEL") or "text-embedding-004"
        dimensionality = config.vertex_embedding_dim
        if dimensionality is None:
            env_dim = os.getenv("VERTEX_AI_EMBEDDING_DIM")
            if env_dim:
                try:
                    dimensionality = int(env_dim)
                except ValueError:
                    print(f"[WARN] Ignoring invalid VERTEX_AI_EMBEDDING_DIM value: {env_dim}", file=sys.stderr)
        return maybe_compute_embeddings_vertex_ai(
            sections,
            project=project,
            location=location,
            model_name=resolved_model,
            dimensionality=dimensionality,
            normalize=config.embedding_normalize,
        )

    if provider == "gemini":
        return maybe_compute_embeddings_gemini(
            sections,
            api_key=config.gemini_api_key,
            model_name=model_name or "gemini-embedding-001",
            task_type=config.gemini_task_type,
            normalize=config.embedding_normalize,
            batch_size=config.gemini_batch_size,
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")


def _classify_ccs_for_record(record: Dict[str, object], config: SummariserConfig) -> Dict[str, object]:
    taxonomy_path = config.ccs_taxonomy_path
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"CCS taxonomy XML not found: {taxonomy_path}")

    taxonomy = load_taxonomy(taxonomy_path)
    classifier = CCSClassifier(
        taxonomy,
        embedding_model=None if config.ccs_embedding_model in (None, "", "none") else config.ccs_embedding_model,
    )
    summary_context = summary_to_prompt_text(record)
    if not summary_context.strip():
        print("[WARN] Skipping CCS classification because the summary text is empty.", file=sys.stderr)
        return record

    client = load_openai_client()
    outcome = classifier.classify(
        record,
        client,
        model=config.ccs_model,
        max_concepts=config.ccs_max_concepts,
        top_candidates=config.ccs_top_candidates,
        fallback_candidates=config.ccs_fallback_candidates,
        temperature=config.ccs_temperature,
        max_output_tokens=config.ccs_max_output_tokens,
        summary_text=summary_context,
    )
    updated = outcome.record
    updated.setdefault("ccs_meta", {})
    updated["ccs_meta"]["taxonomy"] = str(taxonomy.source_path or taxonomy_path)
    updated["ccs_meta"]["model"] = config.ccs_model
    updated["ccs_meta"]["predictions"] = outcome.predictions
    updated["ccs_meta"]["candidates"] = [
        {
            "id": candidate.concept.concept_id,
            "path": candidate.concept.full_path_string(),
            "score": candidate.score,
        }
        for candidate in outcome.candidates
    ]
    updated["ccs_meta"]["prompt"] = outcome.prompt
    return updated


def _write_record(job, record: Dict[str, object], pretty: bool) -> None:
    json_text = json.dumps(record, ensure_ascii=False, indent=2 if pretty else None)
    job.summary_path.parent.mkdir(parents=True, exist_ok=True)
    job.summary_path.write_text(json_text + ("\n" if pretty else ""), encoding="utf-8")


def _summarise_job(job, config: SummariserConfig, pretty: bool) -> None:
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
        compute_embeddings=False,
        section_embeddings=config.section_embeddings,
        embedding_model=None,
        embedding_normalize=config.embedding_normalize,
        embedding_provider=config.embedding_provider,
        vertex_project=config.vertex_project,
        vertex_location=config.vertex_location,
        vertex_embedding_model=config.vertex_embedding_model,
        vertex_embedding_dim=config.vertex_embedding_dim,
        gemini_api_key=config.gemini_api_key,
        gemini_embedding_model=config.gemini_embedding_model,
        gemini_task_type=config.gemini_task_type,
        gemini_batch_size=config.gemini_batch_size,
        classify_ccs=False,
        ccs_taxonomy_path=config.ccs_taxonomy_path,
        ccs_model=config.ccs_model,
        ccs_max_concepts=config.ccs_max_concepts,
        ccs_top_candidates=config.ccs_top_candidates,
        ccs_fallback_candidates=config.ccs_fallback_candidates,
        ccs_temperature=config.ccs_temperature,
        ccs_max_output_tokens=config.ccs_max_output_tokens,
        ccs_embedding_model=config.ccs_embedding_model,
        dual_language=config.dual_language,
        flatten_translations=config.flatten_translations,
    )

    _write_record(job, record, pretty)

    if config.compute_embeddings:
        try:
            embeddings = _compute_embeddings_for_record(record, config)
        except Exception as exc:
            print(f"[WARN] Failed to compute embeddings: {exc}", file=sys.stderr)
        else:
            if embeddings:
                record["embeddings"] = embeddings
                _write_record(job, record, pretty)

    if config.classify_ccs:
        try:
            record = _classify_ccs_for_record(record, config)
        except Exception as exc:
            print(f"[WARN] Failed to classify CCS concepts: {exc}", file=sys.stderr)
        else:
            _write_record(job, record, pretty)



def enqueue_jobs(state: PipelineState, pdfs: Sequence[Path], *, output_dir: Path, base_dirs: Sequence[Path], force: bool) -> Tuple[int, int]:
    enqueued = 0
    skipped = 0
    stats = {STATUS_DONE, STATUS_RUNNING}
    for pdf in pdfs:
        summary_path, matched_base, relative = choose_summary_path(pdf, output_dir=output_dir, bases=base_dirs)
        metadata = {
            "source_size": pdf.stat().st_size,
            "source_mtime": pdf.stat().st_mtime,
            "base_root": str(matched_base) if matched_base else None,
            "relative_pdf": str(relative).replace("\\", "/"),
        }
        job_id = state.enqueue(
            pdf,
            summary_path,
            base_root=matched_base,
            relative_pdf=relative,
            metadata=metadata,
            force=force,
        )
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
    workers: int = 1,
) -> Tuple[int, int]:
    processed = 0
    failed = 0
    quota_abort = False

    def handle_failure(job, exc):
        nonlocal failed, quota_abort
        state.mark_failed(job.job_id, str(exc))
        failed += 1
        print(f"[ERROR] {job.pdf_path}: {exc}", file=sys.stderr)
        if isinstance(exc, EmbeddingQuotaExceeded):
            quota_abort = True
            print("[WARN] Gemini quota exhausted. Halting remaining jobs.", file=sys.stderr)

    workers = max(1, workers)

    if workers == 1:
        while True:
            if quota_abort:
                break
            if limit is not None and processed >= limit:
                quota_abort = True
                break
            job = state.next_queued()
            if job is None:
                break
            state.mark_running(job.job_id)
            try:
                _summarise_job(job, config, pretty)
            except Exception as exc:
                handle_failure(job, exc)
                if quota_abort:
                    break
            else:
                state.mark_done(job.job_id)
                processed += 1
                print(f"[OK] {job.pdf_path} -> {job.summary_path}")
        return processed, failed

    in_flight = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            if quota_abort:
                break
            can_launch_more = limit is None or (processed + len(in_flight)) < limit
            while can_launch_more and not quota_abort:
                job = state.next_queued()
                if job is None:
                    break
                state.mark_running(job.job_id)
                future = executor.submit(_summarise_job, job, config, pretty)
                in_flight[future] = job
                can_launch_more = limit is None or (processed + len(in_flight)) < limit
            if not in_flight:
                break
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                job = in_flight.pop(fut)
                try:
                    fut.result()
                except Exception as exc:
                    handle_failure(job, exc)
                else:
                    state.mark_done(job.job_id)
                    processed += 1
                    print(f"[OK] {job.pdf_path} -> {job.summary_path}")
            if limit is not None and processed >= limit:
                quota_abort = True
        # Drain any remaining tasks after abort/limit
        for fut, job in list(in_flight.items()):
            try:
                fut.result()
            except Exception as exc:
                handle_failure(job, exc)
            else:
                state.mark_done(job.job_id)
                if limit is None or processed < limit:
                    processed += 1
                print(f"[OK] {job.pdf_path} -> {job.summary_path}")
            in_flight.pop(fut, None)

    return processed, failed

def handle_run(args: argparse.Namespace) -> int:
    load_env(explicit=args.env_file)
    config_data = load_config_file(args.config)
    output_dir_override: Optional[Path] = None
    if config_data:
        raw_output_dir = config_data.pop("output_dir", None)
        if raw_output_dir not in (None, ""):
            output_dir_override = Path(raw_output_dir)
    pdfs = discover_pdfs(args.pdf or [], args.pdf_dir or [], args.pattern)
    if not pdfs:
        print("[WARN] No PDF files discovered.")
        return 0

    if output_dir_override and args.output_dir == DEFAULT_OUTPUT_DIR:
        output_dir_candidate = output_dir_override.expanduser()
        print(f"[INFO] Output directory overridden by config file: {output_dir_candidate}")
    else:
        output_dir_candidate = args.output_dir.expanduser()

    output_dir = output_dir_candidate.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = args.db.expanduser().resolve()
    state = PipelineState(db_path)
    try:
        base_dirs = [path.expanduser().resolve() for path in (args.pdf_dir or []) if path]
        if not base_dirs:
            base_dirs = resolve_data_roots()
        else:
            # Always consider configured data roots as fallbacks when matching.
            configured_roots = resolve_data_roots()
            seen = {path for path in base_dirs}
            for root in configured_roots:
                if root not in seen:
                    base_dirs.append(root)
                    seen.add(root)
        enqueued, skipped = enqueue_jobs(state, pdfs, output_dir=output_dir, base_dirs=base_dirs, force=args.force)
        print(f"[INFO] Enqueued {enqueued} job(s); skipped {skipped} existing job(s).")

        config = SummariserConfig()
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
        if args.dual_language != config.dual_language:
            overrides["dual_language"] = args.dual_language
        if args.flatten_translations != config.flatten_translations:
            overrides["flatten_translations"] = args.flatten_translations
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
            workers=args.workers,
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
    run_parser.add_argument(
        "--no-dual-language",
        action="store_false",
        dest="dual_language",
        help="Disable English translations in addition to the primary language (default: enabled).",
    )
    run_parser.add_argument(
        "--no-flatten-translations",
        action="store_false",
        dest="flatten_translations",
        help="Disable copying English translation fields into top-level *_en keys.",
    )
    defaults = SummariserConfig()
    run_parser.set_defaults(
        dual_language=defaults.dual_language,
        flatten_translations=defaults.flatten_translations,
    )
    run_parser.add_argument("--embedding-provider", choices=["local", "vertex-ai", "gemini"], help="Override embedding provider.")
    run_parser.add_argument("--disable-embeddings", action="store_true", help="Skip embedding generation.")
    run_parser.add_argument("--disable-ccs", action="store_true", help="Skip ACM CCS classification.")
    run_parser.add_argument("--chunk-size", type=int, help="Override chunk size used for summarisation.")
    run_parser.add_argument(
        "--temperature",
        type=float,
        help="Override sampling temperature (ignored by GPT-5 family models).",
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker threads for summarisation.",
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
