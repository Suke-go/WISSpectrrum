#!/usr/bin/env python3
"""
Helpers for extracting text and metadata from PDF files.

This module introduces an abstraction layer so different extraction backends
such as PyPDF or GROBID can be plugged into the summarisation pipeline without
entangling the rest of the code.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET


class PdfExtractionError(RuntimeError):
    """Raised when a PDF extractor backend can not fulfil a request."""


@dataclass
class PdfExtractionResult:
    text: str
    tei_xml: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    sections: Optional[List["TeiSection"]] = None


@dataclass
class TeiSection:
    heading: Optional[str]
    text: str
    depth: int = 0


class PdfExtractor:
    """Common interface for PDF extraction strategies."""

    name = "base"

    def extract(self, path: Path) -> PdfExtractionResult:
        raise NotImplementedError


class PyPdfExtractor(PdfExtractor):
    """Fallback extractor that relies on the local PyPDF stack."""

    name = "pypdf"

    def extract(self, path: Path) -> PdfExtractionResult:
        pdf_path = path.expanduser()
        if not pdf_path.exists():
            raise PdfExtractionError(f"PDF not found: {pdf_path}")

        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            try:
                from PyPDF2 import PdfReader  # type: ignore
            except ImportError as exc:
                raise PdfExtractionError(
                    'Neither "pypdf" nor "PyPDF2" is available. Install one with "pip install pypdf".'
                ) from exc

        reader = PdfReader(str(pdf_path))
        pages = []
        for idx, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[WARN] Failed to extract text from page {idx}: {exc}", file=sys.stderr)
                text = ""
            stripped = text.strip()
            if stripped:
                pages.append(stripped)
        if not pages:
            fallback_text = self._extract_with_pdfminer(pdf_path)
            if fallback_text:
                print("[INFO] Falling back to pdfminer for text extraction.", file=sys.stderr)
                return PdfExtractionResult(text=fallback_text)
            raise PdfExtractionError("No extractable text found in the PDF.")
        return PdfExtractionResult(text="\n\n".join(pages))

    def _extract_with_pdfminer(self, pdf_path: Path) -> Optional[str]:
        try:
            from pdfminer.high_level import extract_text  # type: ignore
        except ImportError:
            print("[WARN] pdfminer.six is not installed; skipping fallback extraction.", file=sys.stderr)
            return None

        try:
            text = extract_text(str(pdf_path))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] pdfminer failed to extract text: {exc}", file=sys.stderr)
            return None

        stripped = text.strip()
        if not stripped:
            return None
        # Collapse consecutive blank lines to keep output manageable.
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        return "\n\n".join(lines)


class GrobidExtractor(PdfExtractor):
    """Extractor backed by a running GROBID service."""

    name = "grobid"

    def __init__(
        self,
        base_url: str = "http://localhost:8070",
        healthcheck_timeout: float = 60.0,
        request_timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.healthcheck_timeout = max(0.5, float(healthcheck_timeout))
        self.request_timeout = max(1.0, float(request_timeout))
        self._ensure_available()

    def _ensure_available(self) -> None:
        url = f"{self.base_url}/api/isalive"
        try:
            with urllib.request.urlopen(url, timeout=self.healthcheck_timeout):
                return
        except urllib.error.URLError as exc:
            raise PdfExtractionError(
                f"GROBID service not reachable at {url}. Expected a Docker container exposing the service."
            ) from exc

    def extract(self, path: Path) -> PdfExtractionResult:
        pdf_path = path.expanduser()
        if not pdf_path.exists():
            raise PdfExtractionError(f"PDF not found: {pdf_path}")

        tei_xml = self._process_fulltext(pdf_path)
        metadata = self._extract_metadata(tei_xml)
        sections = self._extract_sections(tei_xml)
        text = self._tei_to_plaintext(tei_xml, sections)
        if not text:
            print("[WARN] GROBID returned empty body text; falling back to PyPDF extractor.", file=sys.stderr)
            fallback = PyPdfExtractor().extract(pdf_path)
            text = fallback.text
        return PdfExtractionResult(text=text, tei_xml=tei_xml, metadata=metadata, sections=sections)

    def _process_fulltext(self, pdf_path: Path) -> str:
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise PdfExtractionError(
                'The "requests" package is required for GROBID integration. Install it with "pip install requests".'
            ) from exc

        url = f"{self.base_url}/api/processFulltextDocument"
        with pdf_path.open("rb") as pdf_file:
            files = {
                "input": (pdf_path.name, pdf_file, "application/pdf"),
            }
            data = {
                "consolidateHeader": "1",
                "consolidateCitations": "0",
                "teiCoordinates": "false",
            }
            try:
                response = requests.post(
                    url,
                    files=files,
                    data=data,
                    timeout=self.request_timeout,
                )
            except requests.RequestException as exc:  # type: ignore[attr-defined]
                raise PdfExtractionError(f"GROBID request failed: {exc}") from exc

        if response.status_code != 200:
            raise PdfExtractionError(
                f"GROBID returned status {response.status_code}: {response.text[:200]}"
            )
        text = response.text.strip()
        if not text:
            raise PdfExtractionError("Received empty response from GROBID.")
        return text

    def _tei_to_plaintext(self, tei_xml: str, sections: Optional[List["TeiSection"]]) -> str:
        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError:
            return ""

        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        if sections:
            paragraphs = [section.text.strip() for section in sections if section.text.strip()]
            return "\n\n".join(paragraphs)

        paragraphs: List[str] = []
        for node in root.findall(".//tei:text//tei:body//tei:p", ns):
            snippet = " ".join(tok.strip() for tok in node.itertext())
            if snippet.strip():
                paragraphs.append(" ".join(snippet.split()))
        if not paragraphs:
            body = root.find(".//tei:text//tei:body", ns)
            if body is not None:
                fallback = " ".join(tok.strip() for tok in body.itertext())
                return " ".join(fallback.split())
        return "\n\n".join(paragraphs)

    def _extract_metadata(self, tei_xml: str) -> Dict[str, Any]:
        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError:
            return {}

        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        metadata: Dict[str, Any] = {}

        title_el = root.find(".//tei:teiHeader//tei:titleStmt/tei:title", ns)
        if title_el is not None:
            title_text = " ".join(title_el.itertext()).strip()
            if title_text:
                metadata["title"] = title_text

        author_nodes = root.findall(".//tei:teiHeader//tei:titleStmt//tei:author", ns)
        authors: List[str] = []
        for author in author_nodes:
            pers = author.find(".//tei:persName", ns)
            if pers is None:
                continue
            name = " ".join(piece.strip() for piece in pers.itertext())
            cleaned = " ".join(name.split())
            if cleaned:
                authors.append(cleaned)
        if authors:
            metadata["authors"] = authors

        abstract_el = root.find(".//tei:teiHeader//tei:profileDesc//tei:abstract", ns)
        if abstract_el is not None:
            abstract_text = " ".join(tok.strip() for tok in abstract_el.itertext()).strip()
            if abstract_text:
                metadata["abstract"] = abstract_text

        doi_el = root.find(".//tei:teiHeader//tei:idno[@type='DOI']", ns)
        if doi_el is not None:
            doi_text = doi_el.text.strip() if doi_el.text else ""
            if doi_text:
                metadata["doi"] = doi_text

        date_el = root.find(".//tei:teiHeader//tei:publicationStmt//tei:date", ns)
        if date_el is not None:
            year = date_el.get("when") or (date_el.text or "")
            year = year.strip()
            if year:
                metadata["year"] = year

        pdf_el = root.find(".//tei:teiHeader//tei:publicationStmt//tei:idno[@type='URL']", ns)
        if pdf_el is not None:
            pdf_text = pdf_el.text.strip() if pdf_el.text else ""
            if pdf_text:
                metadata["pdf"] = pdf_text

        return metadata

    def _extract_sections(self, tei_xml: str) -> List["TeiSection"]:
        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError:
            return []

        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        sections: List[TeiSection] = []

        def normalise_text(text: str) -> str:
            return " ".join(text.split())

        def scrape_div(div_el: ET.Element, inherited_heading: Optional[str], depth: int) -> None:
            local_head = div_el.find("./tei:head", ns)
            heading_text = None
            if local_head is not None:
                heading_text = normalise_text(" ".join(local_head.itertext()))
            heading = heading_text or inherited_heading

            paragraphs: List[str] = []
            for child in div_el:
                tag = child.tag.rsplit("}", 1)[-1]
                if tag == "p":
                    para = normalise_text(" ".join(child.itertext()))
                    if para:
                        paragraphs.append(para)
                elif tag == "div":
                    scrape_div(child, heading, depth + 1)

            cleaned = "\n\n".join(paragraphs).strip()
            if cleaned:
                sections.append(TeiSection(heading=heading, text=cleaned, depth=depth))

        body = root.find(".//tei:text//tei:body", ns)
        if body is None:
            return []

        for child in body:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "div":
                scrape_div(child, inherited_heading=None, depth=0)

        return sections


def build_extractor(
    name: str,
    *,
    grobid_url: Optional[str] = None,
    grobid_timeout: Optional[float] = None,
) -> PdfExtractor:
    """Factory returning the requested extractor with sensible defaults."""

    normalized = (name or "pypdf").strip().lower()
    if normalized in ("pypdf", "pypdf2", "pypdf-legacy", "default"):
        return PyPdfExtractor()
    if normalized == "grobid":
        return GrobidExtractor(
            base_url=grobid_url or "http://localhost:8070",
            healthcheck_timeout=grobid_timeout or 60.0,
            request_timeout=grobid_timeout or 60.0,
        )
    raise PdfExtractionError(f"Unsupported extractor '{name}'. Try 'pypdf' or 'grobid'.")
