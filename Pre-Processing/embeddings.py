#!/usr/bin/env python3
"""Embedding helper utilities for PDF summarisation."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from threading import Lock
from typing import Dict, Iterable, List, Optional


try:
    import numpy as _np  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency guard
    _np = None  # type: ignore[assignment]


_SENTENCE_TRANSFORMER_LOCK = Lock()
_VERTEX_MODEL_LOCK = Lock()
_GEMINI_CLIENT_LOCK = Lock()


@lru_cache(maxsize=8)
def _lazy_sentence_transformer(model_name: str):
    """Load and cache SentenceTransformer instances."""
    with _SENTENCE_TRANSFORMER_LOCK:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer(model_name)


@lru_cache(maxsize=16)
def _lazy_vertex_model(project: str, location: str, model_name: str):
    """Load and cache Vertex AI TextEmbeddingModel instances."""
    with _VERTEX_MODEL_LOCK:
        import vertexai  # type: ignore
        from vertexai.preview.language_models import TextEmbeddingModel  # type: ignore

        vertexai.init(project=project, location=location)
        return TextEmbeddingModel.from_pretrained(model_name)


def maybe_compute_embeddings_local(
    sections: Dict[str, str],
    *,
    model_name: str,
    normalize: bool,
) -> Dict[str, object]:
    if _np is None:
        try:
            import numpy as np  # type: ignore  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SystemExit(
                'Embedding generation requires "sentence-transformers". Install it with "pip install sentence-transformers".'
            ) from exc

    texts: List[str] = []
    keys: List[str] = []
    for key, raw in sections.items():
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        keys.append(key)
        texts.append(text)
    if not texts:
        return {}

    try:
        embedder = _lazy_sentence_transformer(model_name)
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            'Embedding generation requires "sentence-transformers". Install it with "pip install sentence-transformers".'
        ) from exc
    vectors = embedder.encode(texts, normalize_embeddings=normalize)

    result: Dict[str, object] = {
        "sections": {},
    }
    for key, vector in zip(keys, vectors):
        serialized = [float(x) for x in vector]
        result["sections"][key] = serialized
        result[key] = serialized  # backward compatibility for existing consumers
    dimension = len(vectors[0])
    result["model"] = model_name
    result["provider"] = "sentence-transformers"
    result["dim"] = dimension
    result["normed"] = normalize
    return result


def maybe_compute_embeddings_vertex_ai(
    sections: Dict[str, str],
    *,
    project: Optional[str],
    location: Optional[str],
    model_name: str,
    dimensionality: Optional[int],
    normalize: bool,
) -> Dict[str, object]:
    if not project:
        raise ValueError("Vertex AI project ID is required when --embedding-provider=vertex-ai.")
    if not location:
        raise ValueError("Vertex AI location is required when --embedding-provider=vertex-ai.")

    texts: List[str] = []
    keys: List[str] = []
    for key, raw in sections.items():
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        keys.append(key)
        texts.append(text)
    if not texts:
        return {}

    try:
        vertexai_client = _lazy_vertex_model(project, location, model_name)
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            'Vertex AI embeddings require the "google-cloud-aiplatform" package. Install it with "pip install google-cloud-aiplatform".'
        ) from exc

    embeddings = vertexai_client.get_embeddings(
        texts,
        output_dimensionality=dimensionality,
    )
    result: Dict[str, object] = {
        "sections": {},
    }
    vectors: List[List[float]] = []
    for embedding in embeddings:
        vector = [float(x) for x in embedding.values]  # type: ignore[attr-defined]
        if normalize:
            norm = math.sqrt(sum(x * x for x in vector))
            if norm > 0:
                vector = [x / norm for x in vector]
        vectors.append(vector)

    dimension = len(vectors[0])
    for key, vector in zip(keys, vectors):
        result["sections"][key] = vector
        result[key] = vector  # backward compatibility

    result["model"] = model_name
    result["provider"] = "vertex-ai"
    result["dim"] = dimension
    result["normed"] = normalize
    if dimensionality:
        result["dim_requested"] = dimensionality
    result["location"] = location
    return result


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    if batch_size <= 0:
        batch_size = len(items) or 1
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def maybe_compute_embeddings_gemini(
    sections: Dict[str, str],
    *,
    api_key: Optional[str],
    model_name: str,
    task_type: str,
    normalize: bool,
    batch_size: int = 32,
) -> Dict[str, object]:
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini embeddings require GEMINI_API_KEY.")

    prepared: List[tuple[str, str]] = []
    for key, raw in sections.items():
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        prepared.append((key, text))
    if not prepared:
        return {}

    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            'Gemini embeddings require the "google-generativeai" package. Install it with "pip install google-generativeai".'
        ) from exc

    with _GEMINI_CLIENT_LOCK:
        genai.configure(api_key=api_key)

    vectors: List[List[float]] = []
    collected_keys: List[str] = []
    keys = [item[0] for item in prepared]
    values = {key: text for key, text in prepared}
    for batch_keys in _batched(keys, batch_size):
        for key in batch_keys:
            text = values[key]
            response = genai.embed_content(model=model_name, content=text, task_type=task_type)
            vector = response.get("embedding")  # type: ignore[assignment]
            if not isinstance(vector, list):
                raise RuntimeError("Gemini embedding response did not contain 'embedding'.")
            numeric = [float(x) for x in vector]
            if normalize:
                norm = math.sqrt(sum(x * x for x in numeric))
                if norm > 0:
                    numeric = [x / norm for x in numeric]
            vectors.append(numeric)
            collected_keys.append(key)

    if not vectors:
        return {}
    dimension = len(vectors[0])
    result: Dict[str, object] = {
        "sections": {},
        "provider": "gemini",
        "model": model_name,
        "dim": dimension,
        "normed": normalize,
    }
    for key, vector in zip(collected_keys, vectors):
        result["sections"][key] = vector
        result[key] = vector
    return result


def clear_embedding_caches() -> None:
    """Reset cached embedding backends (useful for tests)."""
    _lazy_sentence_transformer.cache_clear()
    _lazy_vertex_model.cache_clear()


__all__ = [
    "maybe_compute_embeddings_local",
    "maybe_compute_embeddings_vertex_ai",
    "maybe_compute_embeddings_gemini",
    "clear_embedding_caches",
]
