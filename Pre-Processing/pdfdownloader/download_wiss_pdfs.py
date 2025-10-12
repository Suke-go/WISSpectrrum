#!/usr/bin/env python3
"""Download WISS proceedings PDFs into year-based directories."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit


CSV_PATTERN = re.compile(r"^wiss(?P<year>\d{4})\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PDFs listed in WISS proceedings CSV files."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("WISSProceedings"),
        help="Directory containing WISS CSV files (default: WISSProceedings).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloads") / "WISS",
        help="Root directory to store downloaded PDFs (default: downloads/WISS).",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        help="Restrict processing to specific years (e.g. --years 2023 2024).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned downloads without saving files.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds (default: 30.0).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between download requests (default: 1.0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files instead of skipping.",
    )
    return parser.parse_args()


def find_csv_files(csv_dir: Path, years: Optional[Iterable[int]]) -> List[Tuple[int, Path]]:
    requested = set(years) if years else None
    files: List[Tuple[int, Path]] = []
    for path in sorted(csv_dir.glob("*.csv")):
        match = CSV_PATTERN.match(path.name)
        if not match:
            continue
        year = int(match.group("year"))
        if requested and year not in requested:
            continue
        files.append((year, path))
    return files


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row]


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


def derive_identifier(row: Dict[str, str], pdf_url: str) -> str:
    raw_id = (row.get("id") or "").strip()
    if raw_id:
        if raw_id.isdigit():
            return f"{int(raw_id):03d}"
        cleaned = slugify(raw_id, fallback="item", max_length=32)
        return cleaned or "item"
    parsed = urlsplit(pdf_url)
    candidate = Path(parsed.path).name
    if candidate.lower().endswith(".pdf"):
        candidate = candidate[:-4]
    candidate = slugify(candidate, fallback="item", max_length=32)
    return candidate or "item"


def build_filename(row: Dict[str, str], pdf_url: str, used_names: Dict[str, int]) -> str:
    identifier = derive_identifier(row, pdf_url)
    title = (row.get("title") or "").strip()
    title_slug = slugify(title, fallback="paper", max_length=60) if title else ""
    stem = identifier if not title_slug else f"{identifier}_{title_slug}"
    counter = used_names[stem]
    if counter:
        filename = f"{stem}_{counter + 1}.pdf"
    else:
        filename = f"{stem}.pdf"
    used_names[stem] += 1
    return filename


def download_file(url: str, destination: Path, *, timeout: float, dry_run: bool, overwrite: bool) -> Optional[str]:
    if destination.exists() and not overwrite:
        return None
    if dry_run:
        return "dry-run"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status and status >= 400:
                return f"HTTP {status}"
            data = response.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - network errors are runtime concerns
        return f"HTTP {exc.code}"
    except Exception as exc:  # pragma: no cover - network errors are runtime concerns
        return f"{exc}"
    destination.write_bytes(data)
    return None


def process_year(
    year: int,
    csv_path: Path,
    output_root: Path,
    *,
    timeout: float,
    delay: float,
    dry_run: bool,
    overwrite: bool,
) -> Tuple[int, int, List[Dict[str, str]]]:
    rows = load_rows(csv_path)
    target_dir = output_root / str(year)
    used_names: Dict[str, int] = defaultdict(int)
    successes = 0
    failures: List[Dict[str, str]] = []
    skipped = 0

    for row in rows:
        pdf_url = (row.get("pdf_url") or "").strip()
        if not pdf_url:
            failures.append(
                {
                    "year": str(year),
                    "title": (row.get("title") or "").strip(),
                    "pdf_url": "",
                    "reason": "missing pdf_url",
                }
            )
            continue

        filename = build_filename(row, pdf_url, used_names)
        destination = target_dir / filename
        outcome = download_file(
            pdf_url,
            destination,
            timeout=timeout,
            dry_run=dry_run,
            overwrite=overwrite,
        )

        if outcome is None:
            successes += 1
            logging.info("[%s] saved %s", year, destination)
            if delay > 0:
                time.sleep(delay)
        elif outcome == "dry-run":
            skipped += 1
            logging.info(
                "[%s] would save %s from %s", year, destination, pdf_url
            )
        else:
            failures.append(
                {
                    "year": str(year),
                    "title": (row.get("title") or "").strip(),
                    "pdf_url": pdf_url,
                    "reason": outcome,
                }
            )
            logging.warning(
                "[%s] failed to download %s -> %s", year, pdf_url, outcome
            )
            if delay > 0 and not dry_run:
                time.sleep(delay)
    total_attempted = successes + skipped
    logging.info(
        "[%s] complete: %s saved, %s planned, %s failures",
        year,
        successes,
        skipped,
        len(failures),
    )
    return successes, len(failures), failures


def write_failures(output_root: Path, failures: List[Dict[str, str]], dry_run: bool) -> None:
    if dry_run or not failures:
        return
    report_path = output_root / "missing_pdfs.csv"
    header = ["year", "title", "pdf_url", "reason"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not report_path.exists()
    with report_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if write_header:
            writer.writeheader()
        for row in failures:
            writer.writerow(row)
    logging.info("Logged %s failures to %s", len(failures), report_path)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


def main() -> int:
    args = parse_args()
    configure_logging()
    if not args.csv_dir.exists():
        logging.error("CSV directory %s does not exist.", args.csv_dir)
        return 1

    files = find_csv_files(args.csv_dir, args.years)
    if not files:
        logging.error("No CSV files found matching criteria.")
        return 1

    total_success = 0
    collected_failures: List[Dict[str, str]] = []

    for year, path in files:
        logging.info("Processing %s...", path)
        success_count, _, failures = process_year(
            year,
            path,
            args.output_dir,
            timeout=args.timeout,
            delay=args.delay,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        total_success += success_count
        collected_failures.extend(failures)

    write_failures(args.output_dir, collected_failures, args.dry_run)
    if collected_failures:
        logging.warning(
            "Completed with %s failures. See log for details.", len(collected_failures)
        )
    logging.info("Downloaded %s files.", total_success)
    return 0


if __name__ == "__main__":
    sys.exit(main())
