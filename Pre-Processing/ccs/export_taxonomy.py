#!/usr/bin/env python3
"""CLI helpers for inspecting the ACM CCS taxonomy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from taxonomy import CCSConcept, CCSTaxonomy, load_taxonomy  # noqa: E402


DEFAULT_XML = THIS_DIR.parent.parent / "ACM CCS" / "acm_ccs2012-1626988337597.xml"


def ensure_taxonomy(xml_path: Path) -> CCSTaxonomy:
    if not xml_path.exists():
        raise SystemExit(f"CCS XML not found at {xml_path}")
    return load_taxonomy(xml_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="Path to the ACM CCS SKOS XML file.")
    parser.add_argument("--output", type=Path, help="Optional path to write JSON output. Defaults to stdout.")
    parser.add_argument("--prompt-catalog", action="store_true", help="Emit a plain-text catalog suitable for LLM prompts.")
    parser.add_argument("--search", metavar="QUERY", help="Search concepts by keywords (case-insensitive).")
    parser.add_argument("--top-k", type=int, default=10, help="Limit for --search results.")
    return parser


def dump_json(taxonomy: CCSTaxonomy, destination: Path | None) -> None:
    payload = taxonomy.to_serializable()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if destination:
        destination.write_text(text, encoding="utf-8")
    else:
        print(text)


def dump_catalog(taxonomy: CCSTaxonomy) -> None:
    for entry in taxonomy.prompt_catalog():
        print(entry)


def run_search(taxonomy: CCSTaxonomy, query: str, top_k: int) -> None:
    results = score_concepts(taxonomy, query)
    for score, concept in results[:top_k]:
        snippet = concept.prompt_snippet()
        print(f"[{score:.3f}] {snippet}")


def score_concepts(taxonomy: CCSTaxonomy, query: str) -> List[Tuple[float, "CCSConcept"]]:
    tokens = [token for token in query.lower().split() if token]
    if not tokens:
        return []
    scored: List[Tuple[float, CCSConcept]] = []
    for concept in taxonomy.concepts():
        haystack = " ".join(
            [
                " ".join(labels)
                for labels, _ in concept.paths
            ]
            + [concept.pref_label]
            + concept.alt_labels
            + ([concept.definition] if concept.definition else [])
            + concept.scope_notes
        ).lower()
        score = sum(haystack.count(token) for token in tokens)
        if score <= 0:
            continue
        scored.append((float(score), concept))
    scored.sort(key=lambda item: (-item[0], item[1].full_path_string()))
    return scored


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    taxonomy = ensure_taxonomy(args.xml)

    if args.search:
        run_search(taxonomy, args.search, args.top_k)
        return 0

    if args.prompt_catalog:
        dump_catalog(taxonomy)
        return 0

    dump_json(taxonomy, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
