#!/usr/bin/env python3
"""Shared command-building utilities for the preprocessing frontends."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

SECTION_CHOICES: Sequence[str] = ("positioning", "purpose", "method", "evaluation", "abstract")
DEFAULT_TAXONOMY = REPO_ROOT / "ACM CCS" / "acm_ccs2012-1626988337597.xml"


@dataclass
class CommandRequest:
    """Normalized representation of a subprocess call."""

    command: List[str]
    cwd: Path
    env: Optional[Dict[str, str]] = None
    cleanup_paths: Optional[List[Path]] = None


@dataclass
class PipelineFormState:
    """Form values collected from a UI for the orchestrator pipeline."""

    pdf_dirs: List[str]
    output_dir: Optional[str] = None
    pattern: Optional[str] = None
    limit: Optional[str] = None
    json_policy: str = "skip"
    env_file: Optional[str] = None
    summary_model: Optional[str] = None
    summary_language: Optional[str] = None
    dual_language: bool = False
    chunk_size: Optional[str] = None
    overlap: Optional[str] = None
    temperature: Optional[str] = None
    chunk_max_tokens: Optional[str] = None
    final_max_tokens: Optional[str] = None
    metadata_chars: Optional[str] = None
    extra_prompt: Optional[str] = None
    extractor: str = "pypdf"
    grobid_url: Optional[str] = None
    grobid_timeout: Optional[str] = None
    compute_embeddings: bool = True
    embedding_provider: Optional[str] = None
    embedding_model_local: Optional[str] = None
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None
    vertex_model: Optional[str] = None
    vertex_dim: Optional[str] = None
    gemini_model: Optional[str] = None
    gemini_task_type: Optional[str] = None
    gemini_batch_size: Optional[str] = None
    embedding_normalize: bool = True
    embedding_sections: Optional[List[str]] = field(default_factory=list)
    embedding_version: Optional[str] = None
    classify_ccs: bool = True
    ccs_model: Optional[str] = None
    ccs_taxonomy: Optional[str] = None
    ccs_max_concepts: Optional[str] = None
    ccs_top_candidates: Optional[str] = None
    ccs_fallback_candidates: Optional[str] = None
    ccs_temperature: Optional[str] = None
    ccs_max_output_tokens: Optional[str] = None
    ccs_embedding_model: Optional[str] = None
    openai_key: Optional[str] = None
    gemini_api_key: Optional[str] = None


@dataclass
class EmbeddingToolOptions:
    """Form values for the embedding recomputation tool."""

    summary_dir: str
    provider: Optional[str] = None
    normalize: bool = False
    sections: Optional[List[str]] = field(default_factory=list)
    embedding_version: Optional[str] = None
    force: bool = False
    dry_run: bool = False
    output_dir: Optional[str] = None
    env_file: Optional[str] = None
    local_model: Optional[str] = None
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None
    vertex_model: Optional[str] = None
    vertex_dim: Optional[str] = None
    gemini_model: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_task_type: Optional[str] = None
    gemini_batch_size: Optional[str] = None
    openai_key: Optional[str] = None


def prepare_pipeline_command(
    options: PipelineFormState,
    python_exec: str,
    repo_root: Path = REPO_ROOT,
) -> Tuple[CommandRequest, Optional[Path]]:
    """Build the orchestrator command and accompanying temp config."""

    if not options.pdf_dirs:
        raise ValueError("At least one PDF directory must be provided.")

    command = [python_exec, "orchestrator.py", "run"]
    for pdf_dir in options.pdf_dirs:
        command += ["--pdf-dir", pdf_dir]

    if options.output_dir:
        command += ["--output-dir", options.output_dir]
    if options.pattern:
        command += ["--pattern", options.pattern]
    if options.limit:
        command += ["--limit", options.limit]
    if options.json_policy == "force":
        command.append("--force")
    if options.env_file:
        command += ["--env-file", options.env_file]
    if options.summary_model:
        command += ["--model", options.summary_model]
    if options.summary_language:
        command += ["--language", options.summary_language]
    if options.dual_language:
        command.append("--dual-language")
    if options.chunk_size:
        command += ["--chunk-size", options.chunk_size]
    if options.temperature:
        command += ["--temperature", options.temperature]
    if options.embedding_provider:
        command += ["--embedding-provider", options.embedding_provider]
    if not options.compute_embeddings:
        command.append("--disable-embeddings")
    if not options.classify_ccs:
        command.append("--disable-ccs")

    config_payload = build_pipeline_config_payload(options)
    config_path: Optional[Path] = None
    if config_payload:
        config_path = write_temp_config(config_payload)
        command += ["--config", str(config_path)]

    env = os.environ.copy()
    if options.openai_key:
        env.setdefault("OPENAI_API_KEY", options.openai_key)
    if options.gemini_api_key:
        env.setdefault("GEMINI_API_KEY", options.gemini_api_key)

    request = CommandRequest(
        command=command,
        cwd=repo_root,
        env=env,
        cleanup_paths=[config_path] if config_path else None,
    )
    return request, config_path


def prepare_embeddings_command(
    options: EmbeddingToolOptions,
    python_exec: str,
    repo_root: Path = REPO_ROOT,
) -> CommandRequest:
    """Build the embedding recomputation command."""

    command = [python_exec, "compute_embeddings.py", options.summary_dir]
    if options.provider:
        command += ["--provider", options.provider]
    if options.normalize:
        command.append("--normalize")
    if options.sections:
        normalized = [section for section in options.sections if section in SECTION_CHOICES]
        if normalized and len(normalized) < len(SECTION_CHOICES):
            command += ["--sections", *normalized]
    if options.embedding_version:
        command += ["--embedding-version", options.embedding_version]
    if options.force:
        command.append("--force")
    if options.dry_run:
        command.append("--dry-run")
    if options.output_dir:
        command += ["--output-dir", options.output_dir]
    if options.env_file:
        command += ["--env-file", options.env_file]

    provider = options.provider or ""
    if provider == "local":
        if options.local_model:
            command += ["--model", options.local_model]
    elif provider == "vertex-ai":
        if options.vertex_project:
            command += ["--vertex-project", options.vertex_project]
        if options.vertex_location:
            command += ["--vertex-location", options.vertex_location]
        if options.vertex_model:
            command += ["--model", options.vertex_model]
        if options.vertex_dim:
            command += ["--vertex-dim", options.vertex_dim]
    elif provider == "gemini":
        if options.gemini_model:
            command += ["--model", options.gemini_model]
        if options.gemini_api_key:
            command += ["--gemini-api-key", options.gemini_api_key]
        if options.gemini_task_type:
            command += ["--gemini-task-type", options.gemini_task_type]
        if options.gemini_batch_size:
            command += ["--gemini-batch-size", options.gemini_batch_size]

    env = os.environ.copy()
    if options.openai_key:
        env.setdefault("OPENAI_API_KEY", options.openai_key)
    if options.gemini_api_key:
        env.setdefault("GEMINI_API_KEY", options.gemini_api_key)

    return CommandRequest(command=command, cwd=repo_root, env=env)


def build_pipeline_config_payload(options: PipelineFormState) -> Dict[str, object]:
    """Translate `PipelineFormState` into orchestrator config overrides."""

    data: Dict[str, object] = {}

    # Summary overrides
    data["chunk_size"] = _coerce_int(options.chunk_size)
    data["overlap"] = _coerce_int(options.overlap)
    data["temperature"] = _coerce_float(options.temperature)
    data["chunk_max_tokens"] = _coerce_int(options.chunk_max_tokens)
    data["final_max_tokens"] = _coerce_int(options.final_max_tokens)
    data["metadata_chars"] = _coerce_int(options.metadata_chars)
    data["dual_language"] = options.dual_language

    if options.extra_prompt:
        data["extra_prompt"] = options.extra_prompt

    extractor = options.extractor
    if extractor:
        data["extractor"] = extractor
        if extractor == "grobid":
            if options.grobid_url:
                data["grobid_url"] = options.grobid_url
            data["grobid_timeout"] = _coerce_float(options.grobid_timeout)

    # Embedding overrides
    data["compute_embeddings"] = options.compute_embeddings
    if options.embedding_provider:
        data["embedding_provider"] = options.embedding_provider
        if options.embedding_provider == "local" and options.embedding_model_local:
            data["embedding_model"] = options.embedding_model_local
        elif options.embedding_provider == "vertex-ai":
            if options.vertex_project:
                data["vertex_project"] = options.vertex_project
            if options.vertex_location:
                data["vertex_location"] = options.vertex_location
            if options.vertex_model:
                data["vertex_embedding_model"] = options.vertex_model
            dim = _coerce_int(options.vertex_dim)
            if dim is not None:
                data["vertex_embedding_dim"] = dim
        elif options.embedding_provider == "gemini":
            if options.gemini_model:
                data["gemini_embedding_model"] = options.gemini_model
            if options.gemini_task_type:
                data["gemini_task_type"] = options.gemini_task_type
            batch_size = _coerce_int(options.gemini_batch_size)
            if batch_size is not None:
                data["gemini_batch_size"] = batch_size
    data["embedding_normalize"] = options.embedding_normalize

    if options.embedding_sections:
        normalized = [section for section in options.embedding_sections if section in SECTION_CHOICES]
        if normalized and len(normalized) < len(SECTION_CHOICES):
            data["embedding_sections"] = normalized

    if options.embedding_version:
        data["embedding_version"] = options.embedding_version

    # CCS overrides
    data["classify_ccs"] = options.classify_ccs
    if options.classify_ccs:
        if options.ccs_model:
            data["ccs_model"] = options.ccs_model
        if options.ccs_taxonomy:
            data["ccs_taxonomy_path"] = options.ccs_taxonomy
        data["ccs_max_concepts"] = _coerce_int(options.ccs_max_concepts)
        data["ccs_top_candidates"] = _coerce_int(options.ccs_top_candidates)
        data["ccs_fallback_candidates"] = _coerce_int(options.ccs_fallback_candidates)
        data["ccs_temperature"] = _coerce_float(options.ccs_temperature)
        data["ccs_max_output_tokens"] = _coerce_int(options.ccs_max_output_tokens)
        if options.ccs_embedding_model:
            data["ccs_embedding_model"] = options.ccs_embedding_model

    return {k: v for k, v in data.items() if v not in (None, "", [], {})}


def write_temp_config(data: Dict[str, object]) -> Path:
    """Persist config overrides to a temporary JSON file."""

    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8")
    json.dump(data, tmp, ensure_ascii=False, indent=2)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _coerce_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _coerce_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


__all__ = [
    "CommandRequest",
    "DEFAULT_TAXONOMY",
    "EmbeddingToolOptions",
    "PipelineFormState",
    "SECTION_CHOICES",
    "build_pipeline_config_payload",
    "prepare_embeddings_command",
    "prepare_pipeline_command",
    "write_temp_config",
]
