"""Parse and query the ACM CCS taxonomy stored as SKOS/RDF XML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
SKOS = "{http://www.w3.org/2004/02/skos/core#}"
SKOSXL = "{http://www.w3.org/2008/05/skos-xl#}"


@dataclass
class CCSConcept:
    """Single ACM CCS concept."""

    concept_id: str
    pref_label: str
    definition: Optional[str] = None
    scope_notes: List[str] = field(default_factory=list)
    alt_labels: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    paths: List[Tuple[List[str], List[str]]] = field(default_factory=list)

    def primary_path_labels(self) -> List[str]:
        if self.paths:
            return self.paths[0][0]
        return [self.pref_label]

    def primary_path_ids(self) -> List[str]:
        if self.paths:
            return self.paths[0][1]
        return [self.concept_id]

    def full_path_string(self, separator: str = " → ") -> str:
        return separator.join(self.primary_path_labels())

    def prompt_snippet(self) -> str:
        parts: List[str] = [self.full_path_string(), f"ID: {self.concept_id}"]
        if self.definition:
            parts.append(f"Definition: {self.definition}")
        if self.scope_notes:
            notes = "; ".join(self.scope_notes)
            parts.append(f"Notes: {notes}")
        if self.alt_labels:
            alts = ", ".join(self.alt_labels)
            parts.append(f"Also known as: {alts}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.concept_id,
            "pref_label": self.pref_label,
            "definition": self.definition,
            "scope_notes": list(self.scope_notes),
            "alt_labels": list(self.alt_labels),
            "parents": list(self.parents),
            "children": list(self.children),
            "paths": [
                {"labels": list(labels), "ids": list(ids)} for labels, ids in self.paths
            ],
        }


class CCSTaxonomy:
    """Convenience wrapper around the parsed CCS concepts."""

    def __init__(
        self,
        *,
        concepts: Dict[str, CCSConcept],
        top_concept_ids: Sequence[str],
        source_path: Optional[Path] = None,
    ) -> None:
        self._concepts = concepts
        self._top_concept_ids = list(top_concept_ids)
        self._source_path = source_path

    @property
    def top_concept_ids(self) -> List[str]:
        return list(self._top_concept_ids)

    @property
    def source_path(self) -> Optional[Path]:
        return self._source_path

    def __contains__(self, concept_id: str) -> bool:
        return concept_id in self._concepts

    def __len__(self) -> int:
        return len(self._concepts)

    def get(self, concept_id: str) -> Optional[CCSConcept]:
        return self._concepts.get(concept_id)

    def concepts(self) -> Iterable[CCSConcept]:
        return self._concepts.values()

    def to_serializable(self) -> List[Dict[str, object]]:
        return [concept.to_dict() for concept in self.concepts()]

    def prompt_catalog(self) -> List[str]:
        return [concept.prompt_snippet() for concept in self.concepts()]


def load_taxonomy(xml: Path) -> CCSTaxonomy:
    return load_taxonomy_from_xml(xml)


def load_taxonomy_from_xml(xml: Path) -> CCSTaxonomy:
    tree = ET.parse(xml)
    root = tree.getroot()

    top_concepts = [
        elem.attrib[RDF + "resource"]
        for elem in root.findall(f"./{SKOS}ConceptScheme/{SKOS}hasTopConcept", {})
    ]

    raw_concepts: Dict[str, CCSConcept] = {}
    pending_children: Dict[str, List[str]] = {}

    for elem in root.findall(f"./{SKOS}Concept"):
        concept_id = elem.attrib[RDF + "about"]
        pref_label = _extract_pref_label(elem)
        definition = _text_or_none(elem.find(f"{SKOS}definition"))
        scope_notes = [
            _normalize_text(note.text)
            for note in elem.findall(f"{SKOS}scopeNote")
            if note.text
        ]
        alt_labels = _extract_alt_labels(elem)
        parents = [
            node.attrib[RDF + "resource"]
            for node in elem.findall(f"{SKOS}broader")
            if RDF + "resource" in node.attrib
        ]
        children = [
            node.attrib[RDF + "resource"]
            for node in elem.findall(f"{SKOS}narrower")
            if RDF + "resource" in node.attrib
        ]

        raw_concepts[concept_id] = CCSConcept(
            concept_id=concept_id,
            pref_label=pref_label,
            definition=definition,
            scope_notes=scope_notes,
            alt_labels=alt_labels,
            parents=parents,
            children=children,
        )
        for child_id in children:
            pending_children.setdefault(child_id, []).append(concept_id)

    # Ensure bidirectional linkage even if <skos:narrower> is missing
    for concept_id, parents in pending_children.items():
        if concept_id in raw_concepts:
            concept = raw_concepts[concept_id]
            for parent_id in parents:
                if parent_id not in concept.parents:
                    concept.parents.append(parent_id)
    for concept in raw_concepts.values():
        for parent_id in concept.parents:
            parent = raw_concepts.get(parent_id)
            if parent is None:
                continue
            if concept.concept_id not in parent.children:
                parent.children.append(concept.concept_id)

    _populate_paths(raw_concepts, top_concepts)
    return CCSTaxonomy(
        concepts=raw_concepts,
        top_concept_ids=top_concepts,
        source_path=xml,
    )


def _extract_pref_label(elem: ET.Element) -> str:
    preferred = elem.find(f"{SKOS}prefLabel")
    if preferred is not None and preferred.text:
        return _normalize_text(preferred.text)
    label = elem.find(f"{SKOSXL}prefLabel")
    if label is not None:
        literal_form = label.find(f".//{SKOSXL}literalForm")
        if literal_form is not None and literal_form.text:
            return _normalize_text(literal_form.text)
    raise ValueError("CCS concept is missing a prefLabel.")


def _extract_alt_labels(elem: ET.Element) -> List[str]:
    alt_labels: List[str] = []
    for label in elem.findall(f"{SKOS}altLabel"):
        if label.text:
            alt_labels.append(_normalize_text(label.text))
    for xl_label in elem.findall(f"{SKOSXL}altLabel"):
        literal_form = xl_label.find(f".//{SKOSXL}literalForm")
        if literal_form is not None and literal_form.text:
            alt_labels.append(_normalize_text(literal_form.text))
    seen = set()
    deduped: List[str] = []
    for entry in alt_labels:
        if entry in seen:
            continue
        seen.add(entry)
        deduped.append(entry)
    return deduped


def _text_or_none(elem: Optional[ET.Element]) -> Optional[str]:
    if elem is not None and elem.text:
        return _normalize_text(elem.text)
    return None


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _populate_paths(concepts: Dict[str, CCSConcept], top_ids: Sequence[str]) -> None:
    from collections import deque

    queue = deque()
    for top_id in top_ids:
        concept = concepts.get(top_id)
        if concept is None:
            continue
        labels = [concept.pref_label]
        ids = [concept.concept_id]
        concept.paths = [(labels, ids)]
        queue.append(concept)

    visited: Dict[str, int] = {}
    while queue:
        current = queue.popleft()
        visited[current.concept_id] = visited.get(current.concept_id, 0) + 1
        for child_id in current.children:
            child = concepts.get(child_id)
            if child is None:
                continue
            new_paths: List[Tuple[List[str], List[str]]] = []
            for labels, ids in current.paths:
                new_labels = labels + [child.pref_label]
                new_ids = ids + [child.concept_id]
                new_paths.append((new_labels, new_ids))
            extended = False
            for labels, ids in new_paths:
                if (labels, ids) not in child.paths:
                    child.paths.append((labels, ids))
                    extended = True
            if extended:
                queue.append(child)
    for concept in concepts.values():
        if concept.paths:
            unique = []
            seen = set()
            for labels, ids in concept.paths:
                key = tuple(ids)
                if key in seen:
                    continue
                seen.add(key)
                unique.append((labels, ids))
            concept.paths = unique
