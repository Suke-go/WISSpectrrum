#!/usr/bin/env python3
"""Embedding helper utilities for PDF summarisation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional


def maybe_compute_embeddings_local(
    sections: Dict[str, str],
    *,
    model_name: str,
    normalize: bool,
) -> Dict[str, object]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np  # type: ignore  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            'Embedding generation requires "sentence-transformers". Install it with "pip install sentence-transformers".'
        ) from exc

    texts: List[str] = []
    keys: List[str] = []
    for key in ("purpose", "method", "evaluation"):
        text = sections.get(key)
        if text and text != "":
            keys.append(key)
            texts.append(text)
    if not texts:
        return {}

    embedder = SentenceTransformer(model_name)
    vectors = embedder.encode(texts, normalize_embeddings=normalize)

    result: Dict[str, object] = {}
    for key, vector in zip(keys, vectors):
        result[key] = [float(x) for x in vector]
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
    try:
        import vertexai  # type: ignore
        from vertexai.preview.language_models import TextEmbeddingModel  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            'Vertex AI embeddings require the "google-cloud-aiplatform" package. Install it with "pip install google-cloud-aiplatform".'
        ) from exc

    if not project:
        raise ValueError("Vertex AI project ID is required when --embedding-provider=vertex-ai.")
    if not location:
        raise ValueError("Vertex AI location is required when --embedding-provider=vertex-ai.")

    texts: List[str] = []
    keys: List[str] = []
    for key in ("purpose", "method", "evaluation"):
        text = sections.get(key)
        if text and text != "":
            keys.append(key)
            texts.append(text)
    if not texts:
        return {}

    vertexai.init(project=project, location=location)
    model = TextEmbeddingModel.from_pretrained(model_name)

    embeddings = model.get_embeddings(
        texts,
        output_dimensionality=dimensionality,
    )
    result: Dict[str, object] = {}
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
        result[key] = vector

    result["model"] = model_name
    result["provider"] = "vertex-ai"
    result["dim"] = dimension
    result["normed"] = normalize
    if dimensionality:
        result["dim_requested"] = dimensionality
    result["location"] = location
    return result


__all__ = [
    "maybe_compute_embeddings_local",
    "maybe_compute_embeddings_vertex_ai",
]
