#!/usr/bin/env python3
"""
Bulk-download PDF files listed in a CSV into a local directory tree.

The CSV is expected to contain at least one column pointing to the PDF URL.
Additional fields (paper id, title, year) are used to build deterministic
filenames when available.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PDFs referenced in a CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV file containing PDF URLs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloads") / "pdfs",
        help="Directory where downloaded PDFs will be stored.",
    )
    parser.add_argument(
        "--url-field",
        default="pdf_url",
        help="CSV column name that contains the PDF URL.",
    )
    parser.add_argument(
        "--id-field",
        default="id",
        help="Optional CSV column that holds a stable identifier.",
    )
    parser.add_argument(
        "--title-field",
        default="title",
        help="Optional CSV column with a human-readable title.",
    )
    parser.add_argument(
        "--year-field",
        default="year",
        help="Optional CSV column used to create subdirectories per year.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout per request (seconds).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between successive downloads (seconds).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned downloads without fetching the files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return [row for row in reader if row]
    except FileNotFoundError as exc:
        raise SystemExit(f"CSV file not found: {csv_path}") from exc
    except Exception as exc:
        raise SystemExit(f"Failed to read CSV at {csv_path}: {exc}") from exc


def slugify(text: str, *, fallback: str, max_length: int) -> str:
    normalized = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    if not normalized:
        normalized = fallback
    if len(normalized) > max_length:
        normalized = normalized[:max_length].rstrip("-")
        if not normalized:
            normalized = fallback
    return normalized


def build_stem(
    row: Dict[str, str],
    *,
    id_field: str,
    title_field: str,
    used_stems: Dict[str, int],
) -> str:
    identifier = (row.get(id_field) or "").strip()
    title = (row.get(title_field) or "").strip()

    if identifier:
        if identifier.isdigit():
            identifier = f"{int(identifier):03d}"
        else:
            identifier = slugify(identifier, fallback="item", max_length=32)
    else:
        identifier = "item"

    if title:
        title_slug = slugify(title, fallback="paper", max_length=60)
        stem = f"{identifier}_{title_slug}"
    else:
        stem = identifier

    counter = used_stems[stem]
    used_stems[stem] += 1
    if counter:
        return f"{stem}_{counter + 1}"
    return stem


def write_bytes(destination: Path, data: bytes, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def download_file(url: str, timeout: float) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
        logging.warning("URL did not look like a PDF: %s (Content-Type=%s)", url, content_type)
    return response.content


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    rows = load_rows(args.csv_path)
    if not rows:
        logging.error("No rows found in %s", args.csv_path)
        return 1

    used_stems: Dict[str, int] = defaultdict(int)
    successes = 0
    skips = 0
    failures: List[str] = []

    for index, row in enumerate(rows, start=1):
        pdf_url = (row.get(args.url_field) or "").strip()
        if not pdf_url:
            failures.append(f"Row {index}: missing {args.url_field}")
            continue

        stem = build_stem(row, id_field=args.id_field, title_field=args.title_field, used_stems=used_stems)
        year = (row.get(args.year_field) or "").strip()
        if year.isdigit():
            target_dir = args.output_dir / year
        else:
            target_dir = args.output_dir
        destination = target_dir / f"{stem}.pdf"

        if destination.exists() and not args.overwrite:
            logging.info("Skipping existing file: %s", destination)
            skips += 1
            continue

        logging.info("Downloading %s -> %s", pdf_url, destination)

        if args.dry_run:
            successes += 1
            continue

        try:
            data = download_file(pdf_url, timeout=args.timeout)
            write_bytes(destination, data, overwrite=args.overwrite)
            successes += 1
        except FileExistsError:
            skips += 1
        except requests.RequestException as exc:
            logging.warning("Failed to download %s: %s", pdf_url, exc)
            failures.append(f"Row {index}: {exc}")
        except Exception as exc:
            logging.error("Unexpected error for %s: %s", pdf_url, exc)
            failures.append(f"Row {index}: {exc}")

        if args.delay > 0 and index != len(rows):
            time.sleep(args.delay)

    logging.info("Finished. downloaded=%s skipped=%s failures=%s", successes, skips, len(failures))
    if failures:
        for note in failures:
            logging.info("Failure detail: %s", note)
    return 0 if successes else 1


if __name__ == "__main__":
    sys.exit(main())

