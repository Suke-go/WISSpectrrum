#!/usr/bin/env python3
"""
Generate a lightweight index for the static summaries viewer.

The script walks through `output/summaries/<year>/*.json` and extracts
basic metadata so the front-end can list years and papers without
loading every JSON up front. It also aggregates ACM CCS concepts so
filters can be applied client-side.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Any, Dict, List

THIS_DIR = Path(__file__).resolve().parent
SUMMARIES_ROOT = THIS_DIR / "summaries"
OUTPUT_PATH = SUMMARIES_ROOT / "index.json"


def _load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_year(name: str) -> int | None:
    try:
        return int(name)
    except ValueError:
        return None


def build_index() -> Dict[str, Any]:
    index: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "years": [],
        "concepts": [],
        "concept_tree": None,
    }

    if not SUMMARIES_ROOT.exists():
        return index

    concepts_map: Dict[str, Dict[str, Any]] = {}

    for year_dir in sorted(SUMMARIES_ROOT.iterdir()):
        if not year_dir.is_dir():
            continue
        year_value = _coerce_year(year_dir.name)
        if year_value is None:
            continue

        papers: List[Dict[str, Any]] = []
        for summary_path in sorted(year_dir.glob("*.json")):
            try:
                data = _load_summary(summary_path)
            except json.JSONDecodeError:
                # Skip malformed JSON but keep the pipeline resilient.
                continue

            relative_path = summary_path.relative_to(SUMMARIES_ROOT)

            # Extract ACM CCS concept information if available.
            ccs = data.get("ccs") or {}
            ccs_ids = ccs.get("ids") or []
            ccs_paths = ccs.get("paths") or []
            explanations = ccs.get("llm_explanations") or []

            concept_refs: List[Dict[str, Any]] = []
            for idx, concept_id in enumerate(ccs_ids):
                if not concept_id:
                    continue
                path = ccs_paths[idx] if idx < len(ccs_paths) else None
                explanation = explanations[idx] if idx < len(explanations) else {}
                confidence = explanation.get("confidence")
                concept_entry = concepts_map.setdefault(
                    concept_id,
                    {
                        "id": concept_id,
                        "path": path,
                        "papers": [],
                    },
                )
                # Update path if previously missing.
                if not concept_entry.get("path") and path:
                    concept_entry["path"] = path

                concept_entry["papers"].append(
                    {
                        "path": str(relative_path).replace("\\", "/"),
                        "slug": summary_path.stem,
                        "title": data.get("title"),
                        "title_en": data.get("title_en"),
                        "year": year_value,
                        "confidence": confidence,
                    }
                )
                concept_refs.append(
                    {
                        "id": concept_id,
                        "path": path,
                        "confidence": confidence,
                    }
                )

            paper_entry: Dict[str, Any] = {
                "path": str(relative_path).replace("\\", "/"),
                "slug": summary_path.stem,
                "title": data.get("title"),
                "title_en": data.get("title_en"),
                "authors": data.get("authors") or [],
                "authors_en": data.get("authors_en") or [],
                "id": data.get("id"),
                "concepts": concept_refs,
            }
            papers.append(paper_entry)

        index["years"].append(
            {
                "year": year_value,
                "papers": papers,
            }
        )

    index["years"].sort(key=lambda item: item["year"])

    concept_list = list(concepts_map.values())
    concept_list.sort(key=lambda item: item.get("path") or item["id"] or "")
    for concept in concept_list:
        concept["papers"].sort(key=lambda paper: (paper["year"], paper["slug"]))
        concept["count"] = len(concept["papers"])
    index["concepts"] = concept_list

    index["concept_tree"] = build_tree_from_paths(concept_list)

    return index


def build_tree_from_paths(concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
    split_pattern = re.compile(r"\s*(?:→|��)\s*")

    root: Dict[str, Any] = {"name": "ACM CCS", "count": 0, "children": {}}

    for concept in concepts:
        path = concept.get("path") or concept.get("id")
        if not path:
            continue

        parts = [part.strip() for part in split_pattern.split(path) if part.strip()]
        if not parts:
            parts = [path.strip()]

        weight = concept.get("count") or len(concept.get("papers") or []) or 1

        current = root
        current["count"] = current.get("count", 0) + weight

        for idx, part in enumerate(parts):
            children = current.setdefault("children", {})
            child = children.get(part)
            if not child:
                child = {"name": part, "count": 0, "children": {}}
                children[part] = child
            child["count"] = child.get("count", 0) + weight
            if idx == len(parts) - 1:
                child["id"] = concept.get("id")
                child["path"] = concept.get("path")
            current = child

    return _compress_children(root)


def _compress_children(node: Dict[str, Any]) -> Dict[str, Any]:
    children_dict = node.get("children")
    if not children_dict:
        node.pop("children", None)
        return node

    children = [_compress_children(child) for child in children_dict.values()]
    children.sort(key=lambda child: child.get("count", 0), reverse=True)
    node["children"] = children
    return node


def main() -> None:
    index = build_index()
    OUTPUT_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
