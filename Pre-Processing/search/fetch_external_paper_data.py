#!/usr/bin/env python3
"""
Fetch metadata for external research papers using publicly documented APIs.

The script can query the OpenAlex API for search results or retrieve specific
records by DOI using either OpenAlex or Crossref. Results are normalised into a
lightweight JSON structure so downstream tooling can ingest them without having
to speak multiple APIs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import quote

import requests

DEFAULT_OUTPUT_PATH = Path("Pre-Processing") / "data" / "external" / "papers.json"
DEFAULT_DELAY = 1.0
DEFAULT_PER_PAGE = 25
MAX_OPENALEX_PER_PAGE = 200
DEFAULT_MAX_RESULTS = 50
DEFAULT_TIMEOUT = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch metadata for research papers from OpenAlex and Crossref."
    )
    parser.add_argument(
        "--query",
        help="Full-text search query (OpenAlex only).",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help=(
            "Additional OpenAlex filter expressions, e.g. "
            "--filter 'concepts.display_name.search:Human-Computer Interaction'. "
            "Can be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--doi",
        action="append",
        help="Fetch one or more DOIs. Can be passed multiple times.",
    )
    parser.add_argument(
        "--doi-file",
        type=Path,
        help="Text file with one DOI per line. Lines starting with # are ignored.",
    )
    parser.add_argument(
        "--source",
        choices=("openalex", "crossref"),
        default="openalex",
        help="API used for DOI lookups (default: openalex).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help="Maximum number of results to return for search queries (default: 50).",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=DEFAULT_PER_PAGE,
        help="Page size for OpenAlex queries (default: 25, max: 200).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Seconds to sleep between API calls to avoid rate limits (default: 1.0).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for results (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--format",
        choices=("json", "ndjson"),
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--email",
        help=(
            "Contact email for the User-Agent header. Providing this is recommended "
            "by both OpenAlex and Crossref."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print fetched records to stdout instead of writing to disk.",
    )
    return parser.parse_args()


def build_user_agent(email: Optional[str]) -> str:
    base = "WISSpectrrumPaperFetcher/0.1"
    if email:
        return f"{base} (mailto:{email})"
    return base


def collect_dois(args: argparse.Namespace) -> List[str]:
    values: List[str] = []
    if args.doi:
        values.extend(args.doi)
    if args.doi_file:
        if not args.doi_file.exists():
            raise FileNotFoundError(f"DOI file not found: {args.doi_file}")
        with args.doi_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                values.append(line)
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        ordered.append(cleaned)
        seen.add(lowered)
    return ordered


def fetch_openalex_query(
    *,
    query: str,
    filters: Sequence[str],
    per_page: int,
    max_results: int,
    delay: float,
    timeout: float,
    headers: Dict[str, str],
) -> List[Dict[str, Any]]:
    url = "https://api.openalex.org/works"
    per_page = max(1, min(per_page, MAX_OPENALEX_PER_PAGE))
    cursor = "*"
    results: List[Dict[str, Any]] = []
    previous_cursor: Optional[str] = None

    while len(results) < max_results:
        params = {
            "search": query,
            "per-page": per_page,
            "cursor": cursor,
        }
        if filters:
            params["filter"] = ",".join(filters)

        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("results") or []
        for item in items:
            results.append(normalise_openalex_record(item))
            if len(results) >= max_results:
                break

        next_cursor = payload.get("meta", {}).get("next_cursor")
        if not next_cursor or next_cursor == previous_cursor:
            break
        previous_cursor = cursor
        cursor = next_cursor
        if len(results) >= max_results:
            break
        time.sleep(max(0.0, delay))

    return results[:max_results]


def fetch_openalex_dois(
    dois: Iterable[str],
    *,
    delay: float,
    timeout: float,
    headers: Dict[str, str],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for doi in dois:
        identifier = doi.strip()
        if identifier.lower().startswith("http"):
            work_path = quote(identifier, safe=":/")
        else:
            work_path = quote(f"https://doi.org/{identifier}", safe=":/")
        url = f"https://api.openalex.org/works/{work_path}"
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as exc:
            print(f"[warn] OpenAlex lookup failed for DOI {doi}: {exc}", file=sys.stderr)
            time.sleep(max(0.0, delay))
            continue
        records.append(normalise_openalex_record(payload))
        time.sleep(max(0.0, delay))
    return records


def fetch_crossref_dois(
    dois: Iterable[str],
    *,
    delay: float,
    timeout: float,
    headers: Dict[str, str],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    session = requests.Session()
    session.headers.update(headers)

    for doi in dois:
        identifier = quote(doi.strip(), safe="")
        url = f"https://api.crossref.org/works/{identifier}"
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as exc:
            print(f"[warn] Crossref lookup failed for DOI {doi}: {exc}", file=sys.stderr)
            time.sleep(max(0.0, delay))
            continue
        except requests.RequestException as exc:
            # Continue on network faults instead of crashing the pipeline.
            print(f"[warn] Crossref lookup error for DOI {doi}: {exc}", file=sys.stderr)
            time.sleep(max(0.0, delay))
            continue
        message = payload.get("message")
        if not message:
            print(f"[warn] Crossref returned no message for DOI {doi}", file=sys.stderr)
            time.sleep(max(0.0, delay))
            continue
        records.append(normalise_crossref_record(message))
        time.sleep(max(0.0, delay))
    return records


def normalise_openalex_record(item: Dict[str, Any]) -> Dict[str, Any]:
    authors: List[Dict[str, Any]] = []
    for authorship in item.get("authorships") or []:
        author = authorship.get("author") or {}
        institutions = [
            inst.get("display_name")
            for inst in authorship.get("institutions") or []
            if inst.get("display_name")
        ]
        authors.append(
            {
                "name": author.get("display_name"),
                "orcid": author.get("orcid"),
                "institutions": institutions or None,
            }
        )

    concepts = [
        concept.get("display_name")
        for concept in item.get("concepts") or []
        if concept.get("display_name")
    ]
    primary_location = item.get("primary_location") or {}
    host_venue = item.get("host_venue") or {}
    open_access = item.get("open_access") or {}

    return {
        "source": "openalex",
        "id": item.get("id"),
        "doi": item.get("doi"),
        "title": item.get("display_name"),
        "publication_year": item.get("publication_year"),
        "publication_date": item.get("publication_date"),
        "type": item.get("type"),
        "language": item.get("language"),
        "abstract": rebuild_openalex_abstract(item.get("abstract_inverted_index")),
        "authorships": authors or None,
        "concepts": concepts or None,
        "primary_location": {
            "source": (primary_location.get("source") or {}).get("display_name"),
            "landing_page_url": primary_location.get("landing_page_url")
            or host_venue.get("url"),
            "is_oa": primary_location.get("is_oa"),
        },
        "host_venue": {
            "display_name": host_venue.get("display_name"),
            "publisher": host_venue.get("publisher"),
            "issn": host_venue.get("issn"),
        },
        "open_access": {
            "status": open_access.get("oa_status"),
            "is_oa": open_access.get("is_oa"),
            "oa_url": open_access.get("oa_url"),
        },
        "cited_by_count": item.get("cited_by_count"),
        "relevance_score": item.get("relevance_score"),
    }


def rebuild_openalex_abstract(abstract_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    if not abstract_index:
        return None
    positions = [pos for occurrences in abstract_index.values() for pos in occurrences]
    if not positions:
        return None
    size = max(positions) + 1
    tokens: List[Optional[str]] = [None] * size
    for word, occurrences in abstract_index.items():
        for pos in occurrences:
            if 0 <= pos < size:
                tokens[pos] = word
    rebuilt = " ".join(token for token in tokens if token)
    return rebuilt or None


def normalise_crossref_record(message: Dict[str, Any]) -> Dict[str, Any]:
    authors_data = message.get("author") or []
    authors: List[Dict[str, Any]] = []
    for entry in authors_data:
        given = entry.get("given") or ""
        family = entry.get("family") or ""
        full_name = " ".join(part for part in (given.strip(), family.strip()) if part)
        authors.append(
            {
                "name": full_name or entry.get("name"),
                "orcid": entry.get("ORCID"),
                "affiliation": [
                    aff.get("name") for aff in entry.get("affiliation") or [] if aff.get("name")
                ]
                or None,
            }
        )

    titles = message.get("title") or []
    title = titles[0] if titles else None
    container_titles = message.get("container-title") or []
    container_title = container_titles[0] if container_titles else None

    published_print = extract_date(message.get("published-print"))
    published_online = extract_date(message.get("published-online"))

    return {
        "source": "crossref",
        "id": message.get("URL"),
        "doi": message.get("DOI"),
        "title": title,
        "abstract": message.get("abstract"),
        "language": message.get("language"),
        "type": message.get("type"),
        "publisher": message.get("publisher"),
        "container_title": container_title,
        "published_print": published_print,
        "published_online": published_online,
        "page": message.get("page"),
        "volume": message.get("volume"),
        "issue": message.get("issue"),
        "author": authors or None,
        "is_referenced_by_count": message.get("is-referenced-by-count"),
        "reference_count": message.get("reference-count"),
        "license": [
            {
                "URL": item.get("URL"),
                "start": extract_date(item.get("start")),
                "delay_in_days": item.get("delay-in-days"),
                "content_version": item.get("content-version"),
            }
            for item in message.get("license") or []
        ]
        or None,
        "link": [
            {
                "URL": item.get("URL"),
                "content_type": item.get("content-type"),
                "intended_application": item.get("intended-application"),
            }
            for item in message.get("link") or []
        ]
        or None,
    }


def extract_date(value: Optional[Dict[str, Any]]) -> Optional[str]:
    if not value:
        return None
    date_parts = value.get("date-parts") or []
    if not date_parts:
        return None
    parts = date_parts[0]
    if not parts:
        return None
    components = [str(part).zfill(2) if idx else str(part) for idx, part in enumerate(parts)]
    return "-".join(components)


def merge_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for record in records:
        key = (record.get("doi") or record.get("id") or "").lower()
        if not key:
            key = f"__anon_{len(merged)}"
        if key in merged:
            merged[key] = prefer_richer_record(merged[key], record)
        else:
            merged[key] = record
    return list(merged.values())


def prefer_richer_record(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    existing_score = record_score(existing)
    incoming_score = record_score(incoming)
    return incoming if incoming_score > existing_score else existing


def record_score(record: Dict[str, Any]) -> int:
    score = 0
    for key in ("abstract", "authorships", "author", "open_access", "link"):
        value = record.get(key)
        if value:
            score += 1
    return score


def write_output(records: List[Dict[str, Any]], *, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "ndjson":
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
    else:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    headers = {"User-Agent": build_user_agent(args.email)}
    timeout = max(1.0, float(args.timeout))
    delay = max(0.0, float(args.delay))

    if not args.query and not (args.doi or args.doi_file):
        print("error: provide --query and/or --doi/--doi-file", file=sys.stderr)
        return 1

    all_records: List[Dict[str, Any]] = []

    if args.query:
        try:
            search_records = fetch_openalex_query(
                query=args.query,
                filters=args.filter,
                per_page=args.per_page,
                max_results=args.max_results,
                delay=delay,
                timeout=timeout,
                headers=headers,
            )
            print(f"[info] Retrieved {len(search_records)} records from OpenAlex search.")
            all_records.extend(search_records)
        except requests.HTTPError as exc:
            print(f"[error] OpenAlex query failed: {exc}", file=sys.stderr)
        except requests.RequestException as exc:
            print(f"[error] OpenAlex query error: {exc}", file=sys.stderr)

    dois = collect_dois(args)
    if dois:
        if args.source == "openalex":
            lookup_records = fetch_openalex_dois(
                dois,
                delay=delay,
                timeout=timeout,
                headers=headers,
            )
        else:
            lookup_records = fetch_crossref_dois(
                dois,
                delay=delay,
                timeout=timeout,
                headers=headers,
            )
        print(f"[info] Retrieved {len(lookup_records)} records from DOI lookups.")
        all_records.extend(lookup_records)

    if not all_records:
        print("[warn] No records fetched.", file=sys.stderr)
        return 1

    merged_records = merge_records(all_records)
    print(f"[info] Consolidated to {len(merged_records)} unique records.")

    if args.dry_run:
        for record in merged_records:
            print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    output_path = args.output.expanduser()
    write_output(merged_records, path=output_path, fmt=args.format)
    print(f"[info] Wrote {len(merged_records)} records to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
