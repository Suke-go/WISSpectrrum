#!/usr/bin/env python3
"""
Build an enhanced ``index.json`` that adds dimensionality-reduced coordinates and
precomputed nearest neighbours for every paper section embedding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from annoy import AnnoyIndex
except ImportError as exc:  # pragma: no cover - convenience guard
    raise SystemExit(
        "Building the enhanced index now requires the 'annoy' package. "
        "Install it with `pip install annoy`."
    ) from exc


SECTIONS: Tuple[str, ...] = (
    "abstract",
    "overview",
    "positioning",
    "purpose",
    "method",
    "evaluation",
)
TOP_NEIGHBOURS = 15
ANNOY_TREES = 64


def load_paper_with_embedding(paper_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load a paper JSON file and extract section embeddings as unit vectors."""
    try:
        with paper_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[WARN] Error loading {paper_path}: {exc}")
        return None

    embeddings_data = data.get("embeddings", {})
    if not isinstance(embeddings_data, dict):
        return None

    result: Dict[str, np.ndarray] = {}
    for section in SECTIONS:
        raw_vector = embeddings_data.get(section)
        if not isinstance(raw_vector, list) or not raw_vector:
            continue
        vector = np.asarray(raw_vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            continue
        result[section] = vector / norm
    return result or None


def compute_reduced_dimensions(vectors: Sequence[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Project normalised embeddings to 2D via PCA (50 -> 2) and t-SNE."""
    if len(vectors) < 2:
        return None, None

    stack = np.vstack(vectors)
    if stack.ndim != 2 or stack.shape[1] == 0:
        return None, None

    target_components = min(50, stack.shape[0], stack.shape[1])
    print(f"Running PCA ({stack.shape[0]}x{stack.shape[1]}) -> {target_components}D")
    pca_50 = PCA(n_components=target_components)
    embeddings_pca50 = pca_50.fit_transform(stack)

    perplexity = min(30, len(vectors) - 1)
    if perplexity < 2:
        perplexity = len(vectors) - 1
    print(
        f"Running t-SNE ({embeddings_pca50.shape[0]}x{embeddings_pca50.shape[1]}) -> "
        f"2D (perplexity={perplexity})"
    )
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_tsne = tsne.fit_transform(embeddings_pca50)

    print(f"Running PCA ({stack.shape[0]}x{stack.shape[1]}) -> 2D")
    pca_2 = PCA(n_components=2)
    embeddings_pca2d = pca_2.fit_transform(stack)

    return embeddings_tsne, embeddings_pca2d


def _annoy_distance_to_cosine(distance: float) -> float:
    """Convert Annoy angular distance back into cosine similarity."""
    cosine = 1.0 - (distance**2) / 2.0
    return max(0.0, min(1.0, float(cosine)))


def compute_ann_neighbours(vectors: Sequence[np.ndarray], top_k: int) -> List[List[Tuple[int, float]]]:
    """Approximate top-k neighbours for each vector using Annoy (angular / cosine space)."""
    if len(vectors) < 2:
        return [[] for _ in vectors]

    dim = vectors[0].shape[0]
    index = AnnoyIndex(dim, metric="angular")
    for idx, vector in enumerate(vectors):
        index.add_item(idx, vector.tolist())
    index.build(ANNOY_TREES)

    neighbours: List[List[Tuple[int, float]]] = []
    for idx in range(len(vectors)):
        raw_indices, raw_distances = index.get_nns_by_item(
            idx,
            top_k + 1,
            include_distances=True,
        )
        bucket: List[Tuple[int, float]] = []
        for neighbour_idx, distance in zip(raw_indices, raw_distances):
            if neighbour_idx == idx:
                continue
            similarity = _annoy_distance_to_cosine(distance)
            bucket.append((neighbour_idx, similarity))
            if len(bucket) >= top_k:
                break
        neighbours.append(bucket)
    return neighbours


def ensure_output_dir(base_dir: Path) -> Tuple[Path, Path]:
    summaries_dir = base_dir / "summaries"
    index_path = summaries_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"index.json not found at {index_path}")
    return summaries_dir, index_path


def main() -> None:
    base_dir = Path(__file__).parent
    summaries_dir, index_path = ensure_output_dir(base_dir)

    print(f"Loading index from {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        index_data = json.load(handle)

    papers_with_embeddings: List[Dict[str, object]] = []
    embeddings_by_section: Dict[str, List[Optional[np.ndarray]]] = {section: [] for section in SECTIONS}

    print("Collecting section embeddings...")
    for year_block in index_data.get("years", []):
        year = year_block.get("year")
        print(f"  Year {year}:")
        for paper in year_block.get("papers", []):
            paper_path = summaries_dir / paper["path"]
            embeddings_dict = load_paper_with_embedding(paper_path)
            if not embeddings_dict:
                continue

            papers_with_embeddings.append({"year": year, "paper": paper})
            paper.pop("embedding_2d", None)
            paper.pop("embedding_neighbors", None)
            for section in SECTIONS:
                embeddings_by_section[section].append(embeddings_dict.get(section))

    print(f"Loaded {len(papers_with_embeddings)} papers with at least one embedding vector.")

    if not papers_with_embeddings:
        print("[WARN] No embeddings available; enhanced index will match the base index.")

    for section in SECTIONS:
        section_vectors = embeddings_by_section[section]
        valid_indices = [idx for idx, vector in enumerate(section_vectors) if vector is not None]
        if len(valid_indices) < 2:
            print(f"Skipping section '{section}': not enough vectors.")
            continue

        valid_vectors = [section_vectors[idx] for idx in valid_indices if section_vectors[idx] is not None]
        print(f"\nProcessing section '{section}' ({len(valid_vectors)} vectors)...")

        tsne_coords, pca_coords = compute_reduced_dimensions(valid_vectors)
        neighbour_lists = compute_ann_neighbours(valid_vectors, TOP_NEIGHBOURS)

        for local_idx, paper_idx in enumerate(valid_indices):
            paper_entry = papers_with_embeddings[paper_idx]["paper"]

            if tsne_coords is not None and pca_coords is not None:
                coords = paper_entry.setdefault("embedding_2d", {})
                coords[section] = {
                    "tsne": tsne_coords[local_idx].astype(float).tolist(),
                    "pca": pca_coords[local_idx].astype(float).tolist(),
                }

            neighbour_payload = []
            for neighbour_local_idx, similarity in neighbour_lists[local_idx]:
                target_paper_idx = valid_indices[neighbour_local_idx]
                target_paper = papers_with_embeddings[target_paper_idx]["paper"]
                neighbour_payload.append(
                    {
                        "slug": target_paper.get("slug"),
                        "similarity": round(similarity, 4),
                    }
                )
            if neighbour_payload:
                neighbours = paper_entry.setdefault("embedding_neighbors", {})
                neighbours[section] = neighbour_payload

    output_path = summaries_dir / "index_enhanced.json"
    print(f"\nWriting enhanced index to {output_path}")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(index_data, handle, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"  Papers in base index: {sum(len(year.get('papers', [])) for year in index_data.get('years', []))}")
    print(f"  Papers with embeddings: {len(papers_with_embeddings)}")
    print(f"  Output file: {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
