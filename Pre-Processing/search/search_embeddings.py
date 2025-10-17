#!/usr/bin/env python3
"""
Perform semantic search over the section-wise embedding index.

Given the JSONL index produced by ``build_embedding_index.py`` this script embeds a
free-text query (using the same backend) and reports the most similar sections.
"""

from __future__ import annotations

import argparse
import json
import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from embeddings import (
    maybe_compute_embeddings_local,
    maybe_compute_embeddings_vertex_ai,
)


@dataclass
class IndexedEmbedding:
    paper_id: Optional[str]
    title: Optional[str]
    year: Optional[int]
    authors: Optional[List[str]]
    section: str
    text: str
    vector: List[float]
    provider: Optional[str]
    model: Optional[str]
    dim: Optional[int]
    normed: bool
    source_path: Optional[str]

    def vector_norm(self) -> float:
        return math.sqrt(sum(x * x for x in self.vector))


def load_index(path: Path) -> List[IndexedEmbedding]:
    records: List[IndexedEmbedding] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            vector = entry.get("embedding")
            if not isinstance(vector, list) or not vector:
                continue
            records.append(
                IndexedEmbedding(
                    paper_id=entry.get("paper_id"),
                    title=entry.get("title"),
                    year=entry.get("year"),
                    authors=entry.get("authors"),
                    section=entry.get("section"),
                    text=entry.get("text", ""),
                    vector=[float(x) for x in vector],
                    provider=entry.get("embedding_provider"),
                    model=entry.get("embedding_model"),
                    dim=entry.get("embedding_dim"),
                    normed=bool(entry.get("embedding_normed")),
                    source_path=entry.get("source_path"),
                )
            )
    return records


SECTION_FIELD_MAP = {
    "positioning": "positioning_summary",
    "purpose": "purpose_summary",
    "method": "method_summary",
    "evaluation": "evaluation_summary",
    "abstract": "abstract",
}


def choose_configuration(
    records: Sequence[IndexedEmbedding],
    *,
    provider: Optional[str],
    model: Optional[str],
) -> Tuple[str, str, Optional[int], bool]:
    combos = sorted(
        {
            (
                rec.provider or "local",
                rec.model or "",
                rec.dim,
                rec.normed,
            )
            for rec in records
        }
    )
    if provider or model:
        matches = [
            combo
            for combo in combos
            if (provider is None or combo[0] == provider)
            and (model is None or combo[1] == model)
        ]
        if not matches:
            available = ", ".join(f"{p}/{m or 'default'}" for p, m, *_ in combos)
            raise ValueError(f"No embeddings matching provider/model. Available: {available}")
        return matches[0]

    if len(combos) > 1:
        formatted = "\n".join(f"- provider={p}, model={m or '(unset)'}" for p, m, *_ in combos)
        raise ValueError(
            "Embedding index contains multiple provider/model combinations.\n"
            "Specify --provider/--model to disambiguate:\n"
            f"{formatted}"
        )

    provider_val, model_val, dim_val, normed = combos[0]
    return provider_val, model_val, dim_val, normed


def embed_query_sections(
    sections: Dict[str, str],
    *,
    provider: str,
    model: str,
    normalize: bool,
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    vertex_dimensionality: Optional[int],
) -> Dict[str, List[float]]:
    if provider == "local":
        result = maybe_compute_embeddings_local(
            sections,
            model_name=model or "intfloat/multilingual-e5-large-instruct",
            normalize=normalize,
        )
    elif provider == "vertex-ai":
        result = maybe_compute_embeddings_vertex_ai(
            sections,
            project=vertex_project,
            location=vertex_location,
            model_name=model or "text-embedding-004",
            dimensionality=vertex_dimensionality,
            normalize=normalize,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    sections_result = result.get("sections")  # type: ignore[assignment]
    if not sections_result:
        raise RuntimeError("Failed to compute query embeddings.")
    vectors: Dict[str, List[float]] = {}
    for key, vector in sections_result.items():
        if key not in sections:
            continue
        vectors[key] = [float(x) for x in vector]  # type: ignore[assignment]
    if not vectors:
        raise RuntimeError("Embedding provider did not return any vectors for the requested sections.")
    return vectors


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return numerator / (norm_a * norm_b)


def rank_matches(
    records: Sequence[IndexedEmbedding],
    query_vector: Sequence[float],
    *,
    require_normed: bool,
) -> List[Tuple[float, IndexedEmbedding]]:
    results: List[Tuple[float, IndexedEmbedding]] = []
    for record in records:
        if require_normed and not record.normed:
            # Skip incompatible entries (should not happen but prevents subtle bugs)
            continue
        if len(record.vector) != len(query_vector):
            continue
        similarity = cosine_similarity(query_vector, record.vector)
        results.append((similarity, record))
    results.sort(key=lambda item: item[0], reverse=True)
    return results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search an embedding index for semantically similar sections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("index", type=Path, help="Path to the JSONL index produced by build_embedding_index.py.")
    parser.add_argument("--query", help="Natural language query to embed (treated as a single section).")
    parser.add_argument("--query-json", type=Path, help="Summary JSON file to reuse as a multi-section query.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of matches to display.")
    parser.add_argument("--provider", help="Force a specific embedding provider (e.g. 'vertex-ai').")
    parser.add_argument("--model", help="Force a specific embedding model (e.g. 'text-embedding-004').")
    parser.add_argument("--vertex-project", help="Google Cloud project ID (required for Vertex AI queries).")
    parser.add_argument("--vertex-location", help="Vertex AI location (defaults to env or 'us-central1').")
    parser.add_argument("--vertex-dim", type=int, help="Optional output dimensionality for Vertex AI query embeddings.")
    parser.add_argument(
        "--exclude-paper",
        action="append",
        help="Paper ID to omit from the results (repeatable). Automatically includes the query JSON's id when provided.",
    )
    return parser.parse_args(argv)


def load_query_sections(query_text: Optional[str], query_json: Optional[Path]) -> Tuple[Dict[str, str], Optional[str]]:
    sections: Dict[str, str] = {}
    query_id: Optional[str] = None
    if query_json:
        data = json.loads(query_json.read_text(encoding="utf-8"))
        raw_id = data.get("id")
        if isinstance(raw_id, str) and raw_id.strip():
            query_id = raw_id.strip()
        for section_name, field in SECTION_FIELD_MAP.items():
            raw = data.get(field)
            if isinstance(raw, str):
                text = raw.strip()
                if text:
                    sections[section_name] = text
    if query_text:
        text = query_text.strip()
        if text:
            sections["query"] = text
    return sections, query_id


def format_match(rank: int, score: float, record: IndexedEmbedding) -> str:
    author_str = ", ".join(record.authors or []) if record.authors else "(authors unknown)"
    title = record.title or "(untitled)"
    paper_id = record.paper_id or "(no id)"
    snippet = record.text.replace("\n", " ")
    if len(snippet) > 240:
        snippet = snippet[:237] + "..."
    return (
        f"[{rank:02d}] score={score:.4f} | {paper_id} | {title} ({record.year or 'n/a'})\n"
        f"     section={record.section} | authors={author_str}\n"
        f"     text={snippet}\n"
        f"     source={record.source_path or '(unknown)'}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    records = load_index(args.index)
    if not records:
        print(f"[ERROR] No embeddings found in {args.index}")
        return 1

    if not args.query and not args.query_json:
        print("[ERROR] Provide --query or --query-json to describe what you are searching for.")
        return 1

    query_sections, inferred_id = load_query_sections(args.query, args.query_json)
    if not query_sections:
        print("[ERROR] Query text was empty after preprocessing.")
        return 1

    excluded_ids = set(args.exclude_paper or [])
    if inferred_id:
        excluded_ids.add(inferred_id)

    try:
        provider, model, dim, normed = choose_configuration(records, provider=args.provider, model=args.model)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    vertex_project = args.vertex_project or None
    vertex_location = args.vertex_location or None
    if provider == "vertex-ai":
        vertex_project = vertex_project or os.getenv("VERTEX_AI_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        vertex_location = vertex_location or os.getenv("VERTEX_AI_LOCATION") or "us-central1"

    if provider == "local":
        vertex_project = None
        vertex_location = None

    try:
        query_vectors = embed_query_sections(
            query_sections,
            provider=provider,
            model=model,
            normalize=normed,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_dimensionality=args.vertex_dim or dim,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to embed query: {exc}")
        return 1

    # Keep only the records for the selected provider/model combo
    filtered_records = [
        rec
        for rec in records
        if (rec.provider or "local") == provider and (rec.model or "") == model and rec.normed == normed
    ]

    top_k = max(1, args.top_k)

    for section_name, vector in query_vectors.items():
        if section_name == "query":
            section_records = filtered_records
        else:
            section_records = [rec for rec in filtered_records if rec.section == section_name]

        if excluded_ids:
            section_records = [
                rec for rec in section_records if not rec.paper_id or rec.paper_id not in excluded_ids
            ]
        if not section_records:
            print(f"[INFO] No index entries for section '{section_name}'.")
            continue
        matches = rank_matches(section_records, vector, require_normed=False)
        if not matches:
            print(f"[INFO] No compatible embeddings found for section '{section_name}'.")
            continue
        print(f"=== Section: {section_name} ===")
        for idx, (score, record) in enumerate(matches[:top_k], start=1):
            print(format_match(idx, score, record))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
