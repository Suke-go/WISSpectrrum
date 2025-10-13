"""Reusable ACM CCS classification helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .taxonomy import CCSConcept, CCSTaxonomy


try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]


@dataclass
class Candidate:
    concept: CCSConcept
    score: float

    def prompt_row(self, rank: int) -> str:
        definition = self.concept.definition or "Not provided."
        notes = "; ".join(self.concept.scope_notes) if self.concept.scope_notes else "None."
        alts = ", ".join(self.concept.alt_labels) if self.concept.alt_labels else "None."
        path = self.concept.full_path_string()
        return (
            f"{rank}. Path: {path}\n"
            f"   ID: {self.concept.concept_id}\n"
            f"   Score: {self.score:.3f}\n"
            f"   Definition: {definition}\n"
            f"   Notes: {notes}\n"
            f"   Also known as: {alts}"
        )


@dataclass
class ClassificationResult:
    record: Dict[str, object]
    predictions: Dict[str, object]
    candidates: List[Candidate]
    prompt: Dict[str, str]


def summary_to_prompt_text(record: Dict[str, object]) -> str:
    parts: List[str] = []
    title = record.get("title")
    if title:
        parts.append(f"Title: {title}")
    abstract = record.get("abstract")
    if abstract and abstract != "Not specified":
        parts.append(f"Abstract: {abstract}")
    for key in ("positioning_summary", "purpose_summary", "method_summary", "evaluation_summary"):
        value = record.get(key)
        if not value or value == "Not specified":
            continue
        label = key.replace("_", " ").title()
        parts.append(f"{label}: {value}")
    return "\n".join(parts)


class CCSClassifier:
    """Classifies summary records into ACM CCS concepts using shortlisted candidates."""

    def __init__(
        self,
        taxonomy: CCSTaxonomy,
        *,
        embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.taxonomy = taxonomy
        self.embedding_model = None if embedding_model in (None, "", "none") else embedding_model
        self._embedder = None
        self._concept_refs: List[CCSConcept] = list(self.taxonomy.concepts())
        self._concept_texts: List[str] = [_concept_embedding_text(concept) for concept in self._concept_refs]
        self._concept_vectors = None
        if self.embedding_model:
            self._ensure_embedder()
            if self._embedder is not None:
                self._concept_vectors = self._embedder.encode(self._concept_texts, normalize_embeddings=True)

    def shortlist(
        self,
        query_text: str,
        *,
        top_k: int,
        fallback_top_k: int,
    ) -> List[Candidate]:
        if self._embedder and self._concept_vectors is not None:
            return _shortlist_with_embeddings(
                self._embedder,
                self._concept_refs,
                self._concept_vectors,
                query_text,
                top_k=top_k,
            )
        return _shortlist_with_keyword_match(
            self._concept_refs,
            self._concept_texts,
            query_text,
            top_k=fallback_top_k,
        )

    def classify(
        self,
        record: Dict[str, object],
        client,
        *,
        model: str,
        max_concepts: int,
        top_candidates: int,
        fallback_candidates: int,
        temperature: float,
        max_output_tokens: int,
        summary_text: Optional[str] = None,
    ) -> ClassificationResult:
        summary_block = summary_text or summary_to_prompt_text(record)
        if not summary_block.strip():
            raise ValueError("Cannot classify CCS concepts without summary text.")
        candidates = self.shortlist(
            summary_block,
            top_k=top_candidates,
            fallback_top_k=fallback_candidates,
        )
        if not candidates:
            raise ValueError("No candidate CCS concepts found for the provided summary.")
        system_prompt, user_prompt = _build_prompt(record, summary_block, candidates, max_concepts=max_concepts)
        raw = call_openai(
            client,
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        predictions = _parse_llm_response(raw)
        enriched = _apply_predictions(
            record,
            self.taxonomy,
            predictions,
            max_concepts=max_concepts,
        )
        return ClassificationResult(
            record=enriched,
            predictions=predictions,
            candidates=candidates,
            prompt={"system": system_prompt, "user": user_prompt},
        )

    def _ensure_embedder(self) -> None:
        if self.embedding_model and self._embedder is None:
            if SentenceTransformer is None:
                raise SystemExit(
                    'Embedding support requires the "sentence-transformers" package. '
                    'Install it with "pip install sentence-transformers".'
                )
            self._embedder = SentenceTransformer(self.embedding_model)


def call_openai(
    client,
    *,
    model: str,
    messages: Sequence[Dict[str, str]],
    max_output_tokens: int,
    temperature: float,
    response_format: Optional[Dict[str, str]] = None,
) -> str:
    response = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        response_format=response_format,
    )
    chunks: List[str] = []
    for item in response.output:
        if item.type != "message":
            continue
        for content in item.content:
            if content.type == "output_text":
                chunks.append(content.text)
    return "".join(chunks).strip()


def _shortlist_with_embeddings(
    embedder,
    concept_refs: Sequence[CCSConcept],
    concept_vectors,
    query_text: str,
    *,
    top_k: int,
) -> List[Candidate]:
    query_vector = embedder.encode([query_text], normalize_embeddings=True)[0]
    scored: List[Candidate] = []
    for vector, concept in zip(concept_vectors, concept_refs):
        score = _cosine_similarity(query_vector, vector)
        scored.append(Candidate(concept=concept, score=float(score)))
    scored.sort(key=lambda candidate: candidate.score, reverse=True)
    return scored[:top_k]


def _shortlist_with_keyword_match(
    concept_refs: Sequence[CCSConcept],
    concept_texts: Sequence[str],
    query_text: str,
    *,
    top_k: int,
) -> List[Candidate]:
    lowered = query_text.lower()
    tokens = [token for token in lowered.split() if token]
    candidates: List[Candidate] = []
    for concept, haystack_text in zip(concept_refs, concept_texts):
        haystack = haystack_text.lower()
        score = 0.0
        if tokens:
            score = sum(haystack.count(token) for token in tokens)
        elif concept.pref_label.lower() in lowered:
            score = 1.0
        if score <= 0.0:
            continue
        candidates.append(Candidate(concept=concept, score=float(score)))
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates[:top_k]


def _concept_embedding_text(concept: CCSConcept) -> str:
    parts = [
        concept.full_path_string(),
        concept.pref_label,
        concept.definition or "",
        " ".join(concept.scope_notes),
        " ".join(concept.alt_labels),
    ]
    return " ".join(part for part in parts if part)


def _cosine_similarity(a, b) -> float:
    dot = float((a * b).sum())
    norm_a = float(math.sqrt((a * a).sum()))
    norm_b = float(math.sqrt((b * b).sum()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_prompt(
    record: Dict[str, object],
    summary_text: str,
    candidates: List[Candidate],
    *,
    max_concepts: int,
) -> Tuple[str, str]:
    system_prompt = (
        "You are an expert indexer who assigns ACM CCS 2012 concepts to computing research papers. "
        "You will receive the paper's title, abstract, and structured summaries. "
        "Use only the provided candidate concepts. "
        f"Select at most {max_concepts} concepts that best describe the paper. "
        "If none of the candidates fit, return an empty list."
    )
    candidate_block = "\n".join(candidate.prompt_row(idx + 1) for idx, candidate in enumerate(candidates))
    metadata_lines = []
    paper_id = record.get("id")
    if paper_id:
        metadata_lines.append(f"Record ID: {paper_id}")
    year = record.get("year")
    if year:
        metadata_lines.append(f"Year: {year}")
    metadata = "\n".join(metadata_lines) or "(no additional metadata)"
    user_prompt = (
        f"{metadata}\n\n"
        "Paper summary:\n"
        f"{summary_text}\n\n"
        "Candidate concepts:\n"
        f"{candidate_block}\n\n"
        "Respond with JSON using the following schema:\n"
        "{\n"
        '  "ccs": [\n'
        '    {\n'
        '      "id": "concept id string",\n'
        '      "path": "concept path string",\n'
        '      "confidence": "high|medium|low",\n'
        '      "justification": "brief reason citing evidence from the paper summary"\n'
        "    }\n"
        "  ],\n"
        '  "rejected": [\n'
        '    {\n'
        '      "id": "concept id string",\n'
        '      "reason": "why it was excluded"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Always use candidate IDs exactly as provided. "
        "Do not invent new concepts. "
        f"Limit the array under \"ccs\" to at most {max_concepts} entries."
    )
    return system_prompt, user_prompt


def _parse_llm_response(raw: str) -> Dict[str, object]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response was not valid JSON: {exc}\nRaw response:\n{raw}") from exc


def _apply_predictions(
    record: Dict[str, object],
    taxonomy: CCSTaxonomy,
    predictions: Dict[str, object],
    *,
    max_concepts: int,
) -> Dict[str, object]:
    chosen: List[Dict[str, str]] = []
    for item in predictions.get("ccs", []):
        concept_id = str(item.get("id", "")).strip()
        if not concept_id or concept_id not in taxonomy:
            continue
        concept = taxonomy.get(concept_id)
        if concept is None:
            continue
        path = concept.full_path_string()
        chosen.append(
            {
                "id": concept_id,
                "path": path,
                "confidence": str(item.get("confidence", "") or "unspecified"),
                "justification": str(item.get("justification", "")).strip(),
            }
        )
        if len(chosen) >= max_concepts:
            break
    updated = dict(record)
    updated.setdefault("ccs", {})
    updated["ccs"]["ids"] = [entry["id"] for entry in chosen]
    updated["ccs"]["paths"] = [entry["path"] for entry in chosen]
    updated["ccs"]["llm_explanations"] = chosen
    updated["ccs"]["rejected"] = predictions.get("rejected", [])
    return updated


__all__ = [
    "CCSClassifier",
    "Candidate",
    "ClassificationResult",
    "call_openai",
    "summary_to_prompt_text",
]
