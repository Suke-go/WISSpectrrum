"""Reusable ACM CCS classification helpers."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .taxonomy import CCSConcept, CCSTaxonomy
except ImportError:  # pragma: no cover - fallback for script execution
    from taxonomy import CCSConcept, CCSTaxonomy  # type: ignore


try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]


_GEMINI_CLIENT_LOCK = Lock()


@dataclass
class Candidate:
    concept: CCSConcept
    score: float
    source: str = "embedding"

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
    translations = record.get("translations")
    english = translations.get("en") if isinstance(translations, dict) else None

    def pick(field: str) -> Optional[object]:
        if isinstance(english, dict):
            value = english.get(field)
            if value not in (None, "", "Not specified"):
                return value
        value = record.get(field)
        if value in ("記述なし",):
            return None
        return value

    title = pick("title")
    if title:
        parts.append(f"Title: {title}")
    authors = pick("authors")
    if isinstance(authors, str):
        author_list = [entry.strip() for entry in authors.split(",") if entry.strip()]
    elif isinstance(authors, list):
        author_list = [str(entry).strip() for entry in authors if str(entry).strip()]
    else:
        author_list = []
    if author_list:
        parts.append(f"Authors: {', '.join(author_list)}")
    year = pick("year") or record.get("year")
    if year:
        parts.append(f"Year: {year}")
    venue = record.get("venue") or record.get("publication") or record.get("published_in")
    if venue:
        parts.append(f"Venue: {venue}")
    keywords = pick("keywords")
    if isinstance(keywords, str):
        keyword_list = [entry.strip() for entry in keywords.split(",") if entry.strip()]
    elif isinstance(keywords, list):
        keyword_list = [str(entry).strip() for entry in keywords if str(entry).strip()]
    else:
        keyword_list = []
    if keyword_list:
        parts.append(f"Keywords: {', '.join(keyword_list[:12])}")
    abstract = pick("abstract")
    if abstract and abstract != "Not specified":
        parts.append(f"Abstract: {abstract}")
    for key in ("positioning_summary", "purpose_summary", "method_summary", "evaluation_summary"):
        value = pick(key)
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
        embedding_model: Optional[str] = "models/text-embedding-004",
        embedding_min_score: float = 0.0,
        embedding_provider: Optional[str] = None,
        embedding_cache_dir: Optional[Path] = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.embedding_provider = _normalize_provider(embedding_model, embedding_provider)
        self.embedding_model = None if embedding_model in (None, "", "none") else embedding_model
        self.embedding_min_score = embedding_min_score
        self.embedding_cache_dir = embedding_cache_dir or (Path(__file__).resolve().parent / ".cache")
        self._embedder = None
        self._concept_refs: List[CCSConcept] = list(self.taxonomy.concepts())
        self._concept_texts: List[str] = [_concept_embedding_text(concept) for concept in self._concept_refs]
        self._concept_vectors = None
        if self.embedding_provider != "none" and self.embedding_model:
            self._ensure_embedder()
            if self._embedder is not None:
                cached_vectors = self._load_cached_concept_vectors()
                if cached_vectors is not None:
                    self._concept_vectors = cached_vectors
                else:
                    print(
                        f"[INFO] Computing CCS concept embeddings via {self.embedding_provider} ({self.embedding_model}). This may take a few minutes...",
                        file=sys.stderr,
                    )
                    vectors = self._embedder.encode(self._concept_texts, normalize_embeddings=True)
                    serialised = [_ensure_float_list(vector) for vector in vectors]
                    self._concept_vectors = serialised
                    self._store_cached_concept_vectors(serialised)

    def shortlist(
        self,
        query_text: str,
        *,
        top_k: int,
        fallback_top_k: int,
    ) -> List[Candidate]:
        if self._embedder and self._concept_vectors is not None:
            candidates = _shortlist_with_embeddings(
                self._embedder,
                self._concept_refs,
                self._concept_vectors,
                query_text,
                top_k=top_k,
            )
            filtered = [candidate for candidate in candidates if candidate.score >= self.embedding_min_score]
            primary = filtered if filtered else candidates
            if len(primary) >= top_k:
                return primary[:top_k]
            fallback = _shortlist_with_keyword_match(
                self._concept_refs,
                self._concept_texts,
                query_text,
                top_k=max(fallback_top_k, top_k),
            )
            return _merge_candidates(primary, fallback, top_k)
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
            if self.embedding_provider == "sentence-transformers":
                if SentenceTransformer is None:
                    raise SystemExit(
                        'Embedding support requires the "sentence-transformers" package. '
                        'Install it with "pip install sentence-transformers".'
                    )
                self._embedder = SentenceTransformer(self.embedding_model)
            elif self.embedding_provider == "gemini":
                self._embedder = _GeminiEmbedder(self.embedding_model)
            else:
                raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")

    def _cache_layout(self) -> Optional[Path]:
        if not self.embedding_model:
            return None
        if not self.embedding_cache_dir:
            return None
        safe_name = _sanitize_model_name(self.embedding_model)
        provider = self.embedding_provider or "none"
        return self.embedding_cache_dir / f"{provider}_{safe_name}.json"

    def _load_cached_concept_vectors(self) -> Optional[List[List[float]]]:
        cache_path = self._cache_layout()
        if cache_path is None or not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        concept_ids = [concept.concept_id for concept in self._concept_refs]
        if payload.get("concept_ids") != concept_ids:
            return None
        vectors = payload.get("vectors")
        if not isinstance(vectors, list):
            return None
        try:
            return [[float(value) for value in vector] for vector in vectors]
        except (TypeError, ValueError):
            return None

    def _store_cached_concept_vectors(self, vectors: List[List[float]]) -> None:
        cache_path = self._cache_layout()
        if cache_path is None:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        payload = {
            "model": self.embedding_model,
            "provider": self.embedding_provider,
            "concept_ids": [concept.concept_id for concept in self._concept_refs],
            "vectors": vectors,
        }
        try:
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass


def _model_supports_temperature(model: str) -> bool:
    lower = (model or "").lower()
    if lower.startswith("gpt-5"):
        return False
    return True


def call_openai(
    client,
    *,
    model: str,
    messages: Sequence[Dict[str, str]],
    max_output_tokens: int,
    temperature: float,
    response_format: Optional[Dict[str, str]] = None,
) -> str:
    attempt_tokens = max_output_tokens
    max_attempts = 6
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        kwargs = {
            "model": model,
            "input": messages,
            "max_output_tokens": attempt_tokens,
        }
        if _model_supports_temperature(model):
            kwargs["temperature"] = temperature
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            response = client.responses.create(**kwargs)
        except TypeError as exc:
            message = str(exc)
            fallback_kwargs = dict(kwargs)
            changed = False
            if "response_format" in fallback_kwargs and "response_format" in message:
                fallback_kwargs.pop("response_format", None)
                changed = True
            if "max_output_tokens" in fallback_kwargs and "max_output_tokens" in message:
                value = fallback_kwargs.pop("max_output_tokens")
                fallback_kwargs["max_tokens"] = value
                changed = True
            if "temperature" in fallback_kwargs and "temperature" in message:
                fallback_kwargs.pop("temperature")
                changed = True
            if not changed:
                raise
            try:
                response = client.responses.create(**fallback_kwargs)
            except TypeError:
                raise exc
        except Exception as exc:
            message = str(exc)
            if "Unsupported parameter" in message and "temperature" in message and "temperature" in kwargs:
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("temperature", None)
                response = client.responses.create(**fallback_kwargs)
            else:
                raise

        chunks: List[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    chunks.append(getattr(content, "text", ""))
        text = "".join(chunks).strip()
        status = getattr(response, "status", "completed")
        reason = getattr(getattr(response, "incomplete_details", None), "reason", None)

        if status != "completed" and reason == "max_output_tokens":
            if attempt_tokens < 2200:
                attempt_tokens = 2200
            else:
                attempt_tokens = min(3000, attempt_tokens + 200)
            print(
                f"[WARN] CCS classification truncated at max_output_tokens; retrying with {attempt_tokens}.",
                file=sys.stderr,
            )
            continue

        if text:
            if status != "completed":
                print(f"[WARN] CCS classification response incomplete ({status}); using partial output.", file=sys.stderr)
            return text

        if attempt < max_attempts:
            print("[WARN] CCS classification response was empty; retrying.", file=sys.stderr)
            continue

        raise ValueError("Received empty response from CCS classification request.")


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
        scored.append(Candidate(concept=concept, score=float(score), source="embedding"))
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
        candidates.append(Candidate(concept=concept, score=float(score), source="keyword"))
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates[:top_k]


def _merge_candidates(
    primary: Sequence[Candidate],
    secondary: Sequence[Candidate],
    top_k: int,
) -> List[Candidate]:
    merged: List[Candidate] = []
    seen: set[str] = set()
    for candidate in primary:
        concept_id = candidate.concept.concept_id
        if concept_id in seen:
            continue
        merged.append(candidate)
        seen.add(concept_id)
        if len(merged) >= top_k:
            return merged
    for candidate in secondary:
        concept_id = candidate.concept.concept_id
        if concept_id in seen:
            continue
        merged.append(candidate)
        seen.add(concept_id)
        if len(merged) >= top_k:
            break
    return merged


class _GeminiEmbedder:
    """Thin wrapper around the Google Gemini embedding endpoint."""

    def __init__(self, model_name: str, *, task_type: str = "retrieval_document") -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("Gemini embeddings require GEMINI_API_KEY to be set in the environment.")
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                'Gemini embeddings require the "google-generativeai" package. Install it with "pip install google-generativeai".'
            ) from exc
        self._client = genai
        self._model_name = model_name
        self._task_type = task_type
        with _GEMINI_CLIENT_LOCK:
            self._client.configure(api_key=api_key)

    def encode(self, texts: Sequence[str], *, normalize_embeddings: bool) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            response = self._client.embed_content(
                model=self._model_name,
                content=text,
                task_type=self._task_type,
            )
            vector = response.get("embedding")  # type: ignore[assignment]
            if not isinstance(vector, list):
                raise RuntimeError("Gemini embedding response did not contain 'embedding'.")
            numeric = [float(value) for value in vector]
            if normalize_embeddings:
                norm = math.sqrt(sum(entry * entry for entry in numeric))
                if norm > 0.0:
                    numeric = [entry / norm for entry in numeric]
            vectors.append(numeric)
        return vectors


def _normalize_provider(model_name: Optional[str], explicit: Optional[str]) -> str:
    if explicit:
        candidate = explicit.strip().lower()
        if candidate in {"gemini", "sentence-transformers", "none"}:
            return candidate
        raise ValueError(f"Unsupported embedding provider override: {explicit}")
    if model_name in (None, "", "none"):
        return "none"
    lowered = str(model_name).lower()
    if lowered.startswith("models/") or "gemini" in lowered or "embedding-gecko" in lowered:
        return "gemini"
    return "sentence-transformers"


def _sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", model_name)


def _ensure_float_list(vector) -> List[float]:
    if isinstance(vector, list):
        return [float(value) for value in vector]
    return [float(value) for value in vector]


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
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for left, right in zip(a, b):
        lx = float(left)
        rx = float(right)
        dot += lx * rx
        norm_a += lx * lx
        norm_b += rx * rx
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


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
        f"Limit the array under \"ccs\" to at most {max_concepts} entries. "
        "Limit the array under \"rejected\" to at most 5 entries. "
        "Keep every justification or reason concise (no more than 120 characters)."
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
