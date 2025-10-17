"""Tests for the CCS classification helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

CCS_DIR = Path(__file__).resolve().parents[1] / "Pre-Processing" / "ccs"
if str(CCS_DIR) not in sys.path:
    sys.path.insert(0, str(CCS_DIR))

from classifier import CCSClassifier, summary_to_prompt_text  # type: ignore  # noqa: E402
from taxonomy import CCSConcept, CCSTaxonomy  # type: ignore  # noqa: E402


class DummyEmbedder:
    """Minimal embedder stub that returns a preset query vector."""

    def __init__(self, query_vector: List[float]) -> None:
        self._query_vector = query_vector

    def encode(self, texts, normalize_embeddings=True):
        # The classifier queries the embedder once per shortlist call with a single string.
        assert len(texts) == 1
        return [self._query_vector]


def make_taxonomy() -> CCSTaxonomy:
    """Create a tiny taxonomy for testing purposes."""
    root = CCSConcept(
        concept_id="root",
        pref_label="Root concept",
        definition="Top level root.",
        children=["child-a", "child-b"],
        paths=[(["Root concept"], ["root"])],
    )
    child_a = CCSConcept(
        concept_id="child-a",
        pref_label="Deep learning",
        definition="Neural network methods.",
        parents=["root"],
        paths=[(["Root concept", "Deep learning"], ["root", "child-a"])],
    )
    child_b = CCSConcept(
        concept_id="child-b",
        pref_label="Embedded systems",
        definition="Hardware oriented systems.",
        parents=["root"],
        paths=[(["Root concept", "Embedded systems"], ["root", "child-b"])],
    )
    concepts = {
        "root": root,
        "child-a": child_a,
        "child-b": child_b,
    }
    return CCSTaxonomy(concepts=concepts, top_concept_ids=["root"])


def test_summary_to_prompt_text_enriches_metadata():
    record = {
        "title": "Sample Paper",
        "authors": ["Alice", "Bob"],
        "year": 2024,
        "venue": "WISS",
        "keywords": ["CCS", "classification"],
        "abstract": "An overview of the system.",
        "positioning_summary": "Research gap description.",
        "purpose_summary": "Goal statement.",
    }
    text = summary_to_prompt_text(record)
    assert "Title: Sample Paper" in text
    assert "Authors: Alice, Bob" in text
    assert "Year: 2024" in text
    assert "Venue: WISS" in text
    assert "Keywords: CCS, classification" in text
    assert "Positioning Summary: Research gap description." in text
    assert "Purpose Summary: Goal statement." in text


def test_shortlist_merges_keyword_candidates_when_embeddings_filtered():
    taxonomy = make_taxonomy()
    classifier = CCSClassifier(taxonomy, embedding_model="none", embedding_min_score=0.9)
    classifier._concept_refs = list(taxonomy.concepts())
    classifier._concept_texts = [
        "Root concept | computing overview",
        "Deep learning | neural network methods",
        "Embedded systems | hardware design",
    ]
    classifier._embedder = DummyEmbedder(query_vector=[0.0, 1.0])
    classifier._concept_vectors = [
        [1.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
    ]

    candidates = classifier.shortlist(
        "This paper presents a neural network for hardware acceleration.",
        top_k=2,
        fallback_top_k=3,
    )

    # Embedding scores fall below the 0.9 threshold, so keyword fallback should step in.
    assert len(candidates) == 2
    assert {candidate.concept.concept_id for candidate in candidates} == {"child-a", "child-b"}
    sources = {candidate.source for candidate in candidates}
    # Expect at least one candidate to come from keyword matching.
    assert "keyword" in sources
