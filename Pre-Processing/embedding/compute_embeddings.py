#!/usr/bin/env python3
"""
Batch compute embeddings for summary JSON files.

Supports local sentence-transformer models, Vertex AI Text Embedding models, and Gemini
embeddings. Updates summaries in-place or writes copies with refreshed embedding blocks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
PREPROCESS_ROOT = THIS_DIR.parent
if str(PREPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_ROOT))

from embeddings import (  # noqa: E402
    EmbeddingQuotaExceeded,
    maybe_compute_embeddings_gemini,
    maybe_compute_embeddings_local,
    maybe_compute_embeddings_vertex_ai,
)
from utils.env import load_env  # noqa: E402
from utils.paths import ensure_preprocess_path  # noqa: E402

ensure_preprocess_path(PREPROCESS_ROOT)


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
            paths.append(path.resolve())
            continue
        if path.is_dir():
            paths.extend(sorted(child.resolve() for child in path.glob("**/*.json")))
            continue
        matches = sorted(Path().glob(raw))
        if matches:
            paths.extend(child.resolve() for child in matches)
            continue
        raise FileNotFoundError(f"No files matched input pattern: {raw}")
    return paths


def load_summary(path: Path) -> Dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON at {path}: {exc}") from exc


def extract_sections(
    record: Dict[str, object],
    *,
    allowed: Optional[Sequence[str]],
    strip_missing_token: str,
) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    candidates = allowed or SECTION_FIELD_MAP.keys()
    for section in candidates:
        field = SECTION_FIELD_MAP.get(section)
        if not field:
            continue
        raw = record.get(field)
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        if not text or text == strip_missing_token:
            continue
        sections[section] = text
    return sections


def compute_embeddings_for_record(
    sections: Dict[str, str],
    *,
    provider: str,
    model: str,
    normalize: bool,
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    vertex_dim: Optional[int],
    gemini_api_key: Optional[str],
    gemini_task_type: str,
    gemini_batch_size: int,
) -> Dict[str, object]:
    if provider == "local":
        return maybe_compute_embeddings_local(
            sections,
            model_name=model,
            normalize=normalize,
        )
    if provider == "vertex-ai":
        return maybe_compute_embeddings_vertex_ai(
            sections,
            project=vertex_project,
            location=vertex_location,
            model_name=model,
            dimensionality=vertex_dim,
            normalize=normalize,
        )
    if provider == "gemini":
        return maybe_compute_embeddings_gemini(
            sections,
            api_key=gemini_api_key,
            model_name=model,
            task_type=gemini_task_type,
            normalize=normalize,
            batch_size=gemini_batch_size,
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")


def should_skip(record: Dict[str, object], *, force: bool) -> bool:
    if force:
        return False
    embeddings = record.get("embeddings")
    if not isinstance(embeddings, dict):
        return False
    sections = embeddings.get("sections")
    if isinstance(sections, dict) and sections:
        return True
    return False


def build_embedding_meta(
    *,
    result: Dict[str, object],
    record: Dict[str, object],
    source_path: Path,
    version: str,
    normalize: bool,
) -> Dict[str, object]:
    created_at = datetime.now(timezone.utc).isoformat()
    sections = result.get("sections")
    section_names = sorted(sections.keys()) if isinstance(sections, dict) else []
    meta: Dict[str, object] = {
        "provider": result.get("provider"),
        "model": result.get("model"),
        "embedding_version": version,
        "normalized": result.get("normed", normalize),
        "dim": result.get("dim"),
        "created_at": created_at,
        "source_summary_id": record.get("id"),
        "source_path": str(source_path),
        "sections": section_names,
    }
    if "location" in result:
        meta["vertex_location"] = result["location"]
    if "dim_requested" in result:
        meta["dim_requested"] = result["dim_requested"]
    if result.get("provider") == "gemini":
        meta["task_type"] = result.get("task_type")
    if section_names:
        meta["section_count"] = len(section_names)
        meta["vector_count"] = len(section_names)
    return meta


def update_record_with_embeddings(
    record: Dict[str, object],
    *,
    result: Dict[str, object],
    meta: Dict[str, object],
    version: str,
) -> None:
    record["embeddings"] = dict(result)
    record["embeddings"]["version"] = version
    record["embedding_meta"] = meta


def determine_missing_token(record: Dict[str, object]) -> str:
    observed: List[str] = []
    for field in SECTION_FIELD_MAP.values():
        raw = record.get(field)
        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped:
                observed.append(stripped)
    for token in ("記述なし", "Not specified"):
        if token in observed:
            return token
    return "Not specified"


def ensure_destination(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_summary(path: Path, record: Dict[str, object], *, dry_run: bool) -> None:
    if dry_run:
        return
    ensure_destination(path)
    text = json.dumps(record, ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def process_file(
    path: Path,
    *,
    args: argparse.Namespace,
) -> Tuple[str, str]:
    record = load_summary(path)
    if should_skip(record, force=args.force):
        return "skipped", f"{path}: embeddings already present"

    missing_token = determine_missing_token(record)
    sections = extract_sections(
        record,
        allowed=args.sections,
        strip_missing_token=missing_token,
    )
    if not sections:
        return "empty", f"{path}: no sections available for embedding"

    started = time.perf_counter()
    try:
        result = compute_embeddings_for_record(
            sections,
            provider=args.provider,
            model=args.model,
            normalize=args.normalize,
            vertex_project=args.vertex_project,
            vertex_location=args.vertex_location,
            vertex_dim=args.vertex_dim,
            gemini_api_key=args.gemini_api_key,
            gemini_task_type=args.gemini_task_type,
            gemini_batch_size=args.gemini_batch_size,
        )
    except EmbeddingQuotaExceeded as exc:
        return "quota", f"{path}: {exc}"
    except Exception as exc:
        return "error", f"{path}: failed to compute embeddings ({exc})"
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if not result or not result.get("sections"):
        return "error", f"{path}: embedding provider returned empty result"

    # augment provider-specific metadata
    if args.provider == "gemini":
        result["task_type"] = args.gemini_task_type

    meta = build_embedding_meta(
        result=result,
        record=record,
        source_path=path,
        version=args.embedding_version,
        normalize=args.normalize,
    )
    meta["latency_ms"] = elapsed_ms

    update_record_with_embeddings(
        record,
        result=result,
        meta=meta,
        version=args.embedding_version,
    )

    if args.output_dir:
        base = getattr(args, "input_root", None)
        relative: Optional[Path] = None
        for candidate in (Path.cwd(), base) if isinstance(base, Path) else (Path.cwd(),):
            if candidate is None:
                continue
            try:
                relative = path.relative_to(candidate)
                break
            except ValueError:
                continue
        if relative is None:
            relative = Path(path.name)
        target = args.output_dir / relative
    else:
        target = path
    save_summary(target, record, dry_run=args.dry_run)
    return "updated", f"{target}: embeddings updated in {elapsed_ms} ms"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch compute embeddings for summary JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Summary JSON files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--provider",
        choices=["local", "vertex-ai", "gemini"],
        default="gemini",
        help="Embedding provider to use.",
    )
    parser.add_argument(
        "--model",
        help="Embedding model identifier for the selected provider. Defaults are provider-specific.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply L2 normalization to embedding vectors before storing.",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        help="Limit embedding computation to specific sections (default: all). Choices: positioning purpose method evaluation abstract.",
    )
    parser.add_argument(
        "--embedding-version",
        default="v1",
        help="Version tag to embed alongside the vectors.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute embeddings even when the summary already contains vectors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory where updated summaries will be written. Defaults to in-place update.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing files.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional .env file to load before running. Defaults to auto-discovery from the current working directory.",
    )

    # Vertex AI specifics
    parser.add_argument("--vertex-project", help="Google Cloud project ID for Vertex AI embeddings.")
    parser.add_argument("--vertex-location", help="Vertex AI location/region (e.g. us-central1).")
    parser.add_argument("--vertex-dim", type=int, help="Optional output dimensionality for Vertex AI embeddings.")

    # Gemini specifics
    parser.add_argument("--gemini-api-key", help="API key to override GEMINI_API_KEY environment variable.")
    parser.add_argument(
        "--gemini-task-type",
        default="SEMANTIC_SIMILARITY",
        help="Task type passed to the Gemini embedding API.",
    )
    parser.add_argument(
        "--gemini-batch-size",
        type=int,
        default=32,
        help="Maximum items per batch when calling Gemini embeddings.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    env_path = load_env(explicit=args.env_file, start=Path.cwd())
    if env_path:
        print(f"[INFO] Loaded environment from {env_path}")

    try:
        inputs = resolve_inputs(args.inputs)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not inputs:
        print("[ERROR] No input files found.")
        return 1

    if args.sections:
        invalid = set(args.sections) - set(SECTION_FIELD_MAP.keys())
        if invalid:
            print(f"[ERROR] Unknown sections requested: {', '.join(sorted(invalid))}")
            return 1

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model is None:
        if args.provider == "gemini":
            args.model = "gemini-embedding-001"
        elif args.provider == "vertex-ai":
            args.model = "text-embedding-005"
        else:
            args.model = "intfloat/multilingual-e5-large-instruct"

    parent_dirs = [path.parent for path in inputs]
    try:
        common_root_str = os.path.commonpath([str(dir_path) for dir_path in parent_dirs])
    except ValueError:
        common_root_str = ""
    args.input_root = Path(common_root_str) if common_root_str else None

    if args.provider == "vertex-ai":
        args.vertex_project = args.vertex_project or os.getenv("VERTEX_AI_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        args.vertex_location = args.vertex_location or os.getenv("VERTEX_AI_LOCATION") or "us-central1"
        if not args.vertex_project:
            print("[ERROR] Vertex AI provider selected but no project ID supplied (use --vertex-project or set VERTEX_AI_PROJECT).")
            return 1

    status_counts = {"updated": 0, "skipped": 0, "empty": 0, "error": 0}
    quota_abort = False
    for path in inputs:
        status, message = process_file(path, args=args)
        status_counts.setdefault(status, 0)
        status_counts[status] += 1
        prefix = {
            "updated": "[OK]",
            "skipped": "[SKIP]",
            "empty": "[WARN]",
            "error": "[ERROR]",
            "quota": "[ERROR]",
        }.get(status, "[INFO]")
        print(f"{prefix} {message}")
        if status == "quota":
            quota_abort = True
            break

    total = sum(status_counts.values())
    if quota_abort:
        remaining = len(inputs) - total
        if remaining > 0:
            print(f"[WARN] Aborted {remaining} pending file(s) after Gemini quota exhaustion.")
    summary_keys = ["updated", "skipped", "empty", "error"]
    if status_counts.get("quota"):
        summary_keys.append("quota")
    summary_parts = ", ".join(f"{key}={status_counts.get(key,0)}" for key in summary_keys)
    print(f"[INFO] Processed {total} file(s): {summary_parts}")
    exit_code = 0
    if status_counts.get("error", 0) > 0 or status_counts.get("quota", 0) > 0:
        exit_code = 1
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
