#!/usr/bin/env python3
"""
Summarise a PDF into structured JSON using the OpenAI Responses API.

Example:
    python summarize_pdf.py ./paper.pdf --output paper.json --pretty --embeddings

The script expects OPENAI_API_KEY to be available via environment variables or a .env file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embeddings import (
    maybe_compute_embeddings_gemini,
    maybe_compute_embeddings_local,
    maybe_compute_embeddings_vertex_ai,
)
from ccs.classifier import CCSClassifier, summary_to_prompt_text
from ccs.taxonomy import load_taxonomy
from pdf_extractors import PdfExtractionError, PdfExtractionResult, TeiSection, build_extractor
from utils.env import load_env

JAPANESE_HEADINGS: Dict[str, str] = {
    "overview": "\u6982\u8981",
    "positioning": "\u7814\u7a76\u306e\u4f4d\u7f6e\u3065\u3051\uff0f\u76ee\u7684",
    "solution": "\u30bd\u30ea\u30e5\u30fc\u30b7\u30e7\u30f3\uff0f\u30a8\u30f3\u30b8\u30cb\u30a2\u30ea\u30f3\u30b0",
    "evaluation": "\u8a55\u4fa1",
}

ENGLISH_HEADINGS: Dict[str, str] = {
    "overview": "Overview",
    "positioning": "Positioning & Purpose",
    "solution": "Solution / Engineering",
    "evaluation": "Evaluation",
}

JAPANESE_MISSING = "\u8a18\u8ff0\u306a\u3057"
ENGLISH_MISSING = "Not specified"

SECTION_CATEGORY_KEYWORDS = {
    "overview": [
        "overview",
        "introduction",
        "background",
        "motivation",
        "summary",
        "はじめに",
        "序論",
        "概要",
        "背景",
    ],
    "positioning": [
        "purpose",
        "goal",
        "motivation",
        "related work",
        "positioning",
        "objective",
        "目的",
        "狙い",
        "課題",
        "位置づけ",
        "関連研究",
        "先行研究",
        "問題設定",
    ],
    "solution": [
        "method",
        "approach",
        "proposed",
        "system",
        "implementation",
        "architecture",
        "手法",
        "方法",
        "提案",
        "システム",
        "設計",
        "仕組み",
    ],
    "evaluation": [
        "evaluation",
        "experiment",
        "results",
        "discussion",
        "user study",
        "考察",
        "評価",
        "実験",
        "結果",
        "検証",
        "ユーザ",
        "調査",
    ],
}


def load_openai_client() -> "OpenAI":  # type: ignore[name-defined]
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - library check
        raise SystemExit(
            "The \"openai\" package is required. Install it with \"pip install openai\"."
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file."
        )
    return OpenAI(api_key=api_key)


def chunk_text(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        snippet = text[start:end]
        if end < length:
            boundary = max(
                snippet.rfind("\n"),
                snippet.rfind("。"),
                snippet.rfind("、"),
                snippet.rfind("."),
                snippet.rfind("!"),
                snippet.rfind("?"),
            )
            if boundary > 200:
                end = start + boundary + 1
                snippet = text[start:end]
        yield snippet.strip()
        if end == length:
            break
        start = max(0, end - overlap)


def chunk_sections(
    sections: Sequence[TeiSection],
    *,
    chunk_size: int,
    overlap: int,
) -> Iterable[str]:
    def classify_heading(heading: Optional[str], depth: int) -> Optional[str]:
        if not heading:
            return None
        normalized = heading.lower()
        for category, keywords in SECTION_CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in heading or keyword in normalized:
                    return category
        if depth > 0:
            return None
        return None

    def format_chunk(heading: Optional[str], body: str, depth: int) -> str:
        title = heading.strip() if heading else ""
        category = classify_heading(heading, depth)
        prefix_lines: List[str] = []
        if title:
            prefix_lines.append(f"[Section] {title}")
        if category:
            prefix_lines.append(f"[Category] {category}")
        prefix = "\n".join(prefix_lines)
        if prefix:
            return f"{prefix}\n\n{body.strip()}"
        return body.strip()

    for section in sections:
        heading = section.heading
        body = section.text.strip()
        if not body:
            continue
        if len(body) <= chunk_size:
            yield format_chunk(heading, body, section.depth)
            continue
        for chunk in chunk_text(body, chunk_size=chunk_size, overlap=overlap):
            if not chunk:
                continue
            yield format_chunk(heading, chunk, section.depth)


def section_labels_for(language: str) -> Dict[str, str]:
    lower = language.lower()
    if lower.startswith(("ja", "nihon", "jp", "japanese", "\u65e5\u672c")):
        return JAPANESE_HEADINGS
    return ENGLISH_HEADINGS


def missing_text_token(language: str) -> str:
    lower = language.lower()
    if lower.startswith(("ja", "nihon", "jp", "japanese", "\u65e5\u672c")):
        return JAPANESE_MISSING
    return ENGLISH_MISSING


def call_openai(client: "OpenAI", *, model: str, messages: Sequence[Dict[str, str]], temperature: float, max_output_tokens: int) -> str:
    response = client.responses.create(
        model=model,
        input=list(messages),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return response.output_text.strip()


def strip_json_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1]
        if "```" in value:
            value = value.rsplit("```", 1)[0]
    return value.strip()


def parse_json_response(text: str) -> Dict[str, object]:
    value = strip_json_fence(text)
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from model response: {exc}: {value}")


def extract_metadata(client: "OpenAI", *, model: str, excerpt: str, temperature: float, max_tokens: int) -> Dict[str, object]:
    if not excerpt.strip():
        return {}
    system_prompt = (
        "You extract bibliographic metadata from academic papers. "
        "Respond with strict JSON only."
    )
    user_prompt = (
        "Extract metadata from the following excerpt. Return JSON with keys: "
        "title (string or null), authors (array of strings), abstract (string or null), "
        "doi (string or null), year (integer or null)."
        "\n\nExcerpt:\n"
        f"{excerpt.strip()}"
    )
    raw = call_openai(
        client,
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    try:
        return parse_json_response(raw)
    except ValueError:
        print("[WARN] Metadata extraction returned non-JSON response; ignoring.", file=sys.stderr)
        return {}


def summarize_chunk(
    client: "OpenAI",
    *,
    model: str,
    language: str,
    content: str,
    headings: Dict[str, str],
    missing_token: str,
    max_output_tokens: int,
    temperature: float,
) -> str:
    system_prompt = (
        "You are a research assistant who writes clear, literal notes without metaphors."
    )

    overview_label = headings["overview"]
    positioning_label = headings["positioning"]
    solution_label = headings["solution"]
    evaluation_label = headings["evaluation"]

    heading_lines = "\n".join(
        f"- {label}:" for label in (overview_label, positioning_label, solution_label, evaluation_label)
    )
    guidance_lines = "\n".join(
        [
            f"- {overview_label}: Summarise the domain context and why the topic matters in 1-2 sentences.",
            f"- {positioning_label}: Describe the concrete problem, gap, or objective the work targets and who is affected.",
            f"- {solution_label}: Outline the proposed approach or system, highlighting the novel engineering or research contribution.",
            f"- {evaluation_label}: Explain how the work was evaluated (methodology, participants, measures) and note the key outcomes or insights.",
        ]
    )
    user_prompt = (
        f"You will get a portion of a research paper. Using only that portion, "
        f"write concise bullet notes in {language}. Group the notes under the headings listed below. "
        f"Each heading should have 1-2 bullets, and each bullet should contain 1-2 sentences with concrete details. "
        f"If the excerpt lacks information for a heading, write '{missing_token}'."
        "\n\nHeadings:\n"
        f"{heading_lines}\n\n"
        "Guidance:\n"
        f"{guidance_lines}\n\n"
        "Excerpt:\n"
        f"{content}"
    )
    return call_openai(
        client,
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def synthesize_record(
    client: "OpenAI",
    *,
    model: str,
    language: str,
    chunk_notes: Sequence[str],
    metadata_hints: Dict[str, object],
    headings: Dict[str, str],
    missing_token: str,
    max_output_tokens: int,
    temperature: float,
) -> Dict[str, object]:
    system_prompt = (
        "You are a research assistant who produces structured JSON summaries. "
        "Keep the tone literal and avoid metaphors."
    )
    hint_lines = []
    for key, value in metadata_hints.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            display = ", ".join(str(item) for item in value)
        else:
            display = str(value)
        hint_lines.append(f"- {key}: {display}")
    hints_block = "\n".join(hint_lines) or "(no explicit hints)"
    notes_block = "\n\n".join(
        f"Chunk {idx + 1} notes:\n{note}" for idx, note in enumerate(chunk_notes)
    )
    schema_description = json.dumps(
        {
            "title": "string or null",
            "authors": "array of strings",
            "abstract": "string",
            "positioning_summary": "string",
            "purpose_summary": "string",
            "method_summary": "string",
            "evaluation_summary": "string",
            "doi": "string or null",
            "year": "integer or null",
            "ccs_paths": "array of strings",
            "ccs_ids": "array of strings"
        },
        ensure_ascii=False,
        indent=2,
    )
    user_prompt = (
        f"Generate a JSON object that follows the schema below. "
        f"Write all textual fields in {language} with clear, literal sentences. Avoid metaphors and excessive jargon. "
        f"For each *_summary field, write 4 sentences (maximum 6). The first sentence should restate the central idea, and the second sentence should mention specific details such as targets, mechanisms, participants, or quantitative outcomes when available. "
        f"For {headings['evaluation']} specifically, ensure the sentences cover both the evaluation methodology and the observed results. "
        f"If information is unavailable, output '{missing_token}' for summaries and null/[] for other fields as appropriate."
        "\n\nSchema:\n"
        f"{schema_description}\n\n"
        "Metadata hints:\n"
        f"{hints_block}\n\n"
        "Chunk notes:\n"
        f"{notes_block}\n\n"
        "Return JSON only."
    )
    raw = call_openai(
        client,
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return parse_json_response(raw)


def prefer_cli(cli_value, extracted_value):
    return cli_value if cli_value not in (None, "", [], {}) else extracted_value


YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def coerce_year(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = YEAR_PATTERN.search(value)
        if match:
            return int(match.group())
        return None
    if isinstance(value, Iterable):
        for item in value:
            parsed = coerce_year(item)
            if parsed:
                return parsed
    return None


def ensure_list(value: object) -> List[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()]


def summarise_pdf(
    pdf_path: Path,
    *,
    extractor: str,
    grobid_url: Optional[str],
    grobid_timeout: float,
    model: str,
    language: str,
    chunk_size: int,
    overlap: int,
    temperature: float,
    chunk_max_tokens: int,
    final_max_tokens: int,
    metadata_chars: int,
    paper_id: Optional[str],
    title: Optional[str],
    authors: Optional[List[str]],
    year: Optional[int],
    pdf_link: Optional[str],
    code_link: Optional[str],
    ccs_paths_cli: Optional[List[str]],
    ccs_ids_cli: Optional[List[str]],
    compute_embeddings: bool,
    embedding_model: str,
    embedding_normalize: bool,
    embedding_provider: str,
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    vertex_embedding_model: str,
    vertex_embedding_dim: Optional[int],
    gemini_api_key: Optional[str],
    gemini_embedding_model: str,
    gemini_task_type: str,
    gemini_batch_size: int,
    classify_ccs: bool,
    ccs_taxonomy_path: Path,
    ccs_model: str,
    ccs_max_concepts: int,
    ccs_top_candidates: int,
    ccs_fallback_candidates: int,
    ccs_temperature: float,
    ccs_max_output_tokens: int,
    ccs_embedding_model: Optional[str],
) -> Dict[str, object]:
    try:
        extractor_backend = build_extractor(
            extractor,
            grobid_url=grobid_url,
            grobid_timeout=grobid_timeout,
        )
    except PdfExtractionError as exc:
        raise ValueError(str(exc)) from exc

    try:
        extraction: PdfExtractionResult = extractor_backend.extract(pdf_path)
    except PdfExtractionError as exc:
        raise ValueError(str(exc)) from exc

    text = extraction.text
    sections = extraction.sections or []
    extraction_metadata: Dict[str, object] = dict(extraction.metadata or {})
    client = load_openai_client()

    excerpt = text[:metadata_chars]
    metadata_llm = extract_metadata(
        client,
        model=model,
        excerpt=excerpt,
        temperature=max(0.0, min(temperature, 0.5)),
        max_tokens=400,
    )

    headings = section_labels_for(language)
    missing_token = missing_text_token(language)

    if sections:
        chunks = list(
            chunk_sections(
                sections,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    else:
        chunks = list(chunk_text(text, chunk_size=chunk_size, overlap=overlap))
    if not chunks:
        raise ValueError("Failed to create text chunks for summarisation.")

    chunk_notes: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"[INFO] Summarising chunk {idx}/{len(chunks)} ({len(chunk)} chars)...", file=sys.stderr)
        note = summarize_chunk(
            client,
            model=model,
            language=language,
            content=chunk,
            headings=headings,
            missing_token=missing_token,
            max_output_tokens=chunk_max_tokens,
            temperature=temperature,
        )
        chunk_notes.append(note)

    metadata_hints: Dict[str, object] = {
        "id": paper_id,
        "title_hint": title or extraction_metadata.get("title") or metadata_llm.get("title"),
        "authors_hint": authors or extraction_metadata.get("authors") or metadata_llm.get("authors"),
        "doi": extraction_metadata.get("doi") or metadata_llm.get("doi"),
        "year_hint": year or extraction_metadata.get("year") or metadata_llm.get("year"),
        "pdf_link": pdf_link,
        "code_link": code_link,
        "ccs_paths_hint": ccs_paths_cli,
        "ccs_ids_hint": ccs_ids_cli,
    }

    structured = synthesize_record(
        client,
        model=model,
        language=language,
        chunk_notes=chunk_notes,
        metadata_hints=metadata_hints,
        headings=headings,
        missing_token=missing_token,
        max_output_tokens=final_max_tokens,
        temperature=temperature,
    )

    record: Dict[str, object] = {}
    record["id"] = paper_id or structured.get("doi") or extraction_metadata.get("doi") or metadata_llm.get("doi")
    record["title"] = prefer_cli(
        title,
        structured.get("title") or extraction_metadata.get("title") or metadata_llm.get("title"),
    )

    authors_value = prefer_cli(
        authors,
        structured.get("authors") or extraction_metadata.get("authors") or metadata_llm.get("authors"),
    )
    if isinstance(authors_value, str):
        authors_list = [a.strip() for a in authors_value.split(",") if a.strip()]
    else:
        authors_list = list(authors_value or [])
    record["authors"] = authors_list

    record["abstract"] = (
        structured.get("abstract") or extraction_metadata.get("abstract") or metadata_llm.get("abstract") or missing_token
    )
    record["positioning_summary"] = structured.get("positioning_summary") or missing_token
    record["purpose_summary"] = structured.get("purpose_summary") or missing_token
    record["method_summary"] = structured.get("method_summary") or missing_token
    record["evaluation_summary"] = structured.get("evaluation_summary") or missing_token

    chosen_year = prefer_cli(
        year,
        structured.get("year") or extraction_metadata.get("year") or metadata_llm.get("year"),
    )
    record["year"] = coerce_year(chosen_year)

    ccs_paths = ensure_list(prefer_cli(ccs_paths_cli, structured.get("ccs_paths")))
    ccs_ids = ensure_list(prefer_cli(ccs_ids_cli, structured.get("ccs_ids")))
    record["ccs"] = {"paths": ccs_paths, "ids": ccs_ids}

    record["links"] = {
        "pdf": pdf_link or structured.get("pdf") or extraction_metadata.get("pdf") or metadata_llm.get("pdf"),
        "code": code_link or structured.get("code") or extraction_metadata.get("code") or metadata_llm.get("code"),
    }

    if compute_embeddings:
        def _usable_section(value: object) -> Optional[str]:
            if value in (None, "", [], {}):
                return None
            text = str(value).strip()
            if not text or text == missing_token:
                return None
            return text

        sections = {}
        candidates = {
            "positioning": record.get("positioning_summary"),
            "purpose": record.get("purpose_summary"),
            "method": record.get("method_summary"),
            "evaluation": record.get("evaluation_summary"),
            "abstract": record.get("abstract"),
        }
        for key, candidate in candidates.items():
            normalized = _usable_section(candidate)
            if normalized:
                sections[key] = normalized
        if embedding_provider == "local":
            embeddings = maybe_compute_embeddings_local(
                sections,
                model_name=embedding_model,
                normalize=embedding_normalize,
            )
        elif embedding_provider == "vertex-ai":
            resolved_project = vertex_project or os.getenv("VERTEX_AI_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
            resolved_location = vertex_location or os.getenv("VERTEX_AI_LOCATION") or "us-central1"
            resolved_model = vertex_embedding_model or os.getenv("VERTEX_AI_EMBEDDING_MODEL") or "text-embedding-004"
            resolved_dim = vertex_embedding_dim
            if resolved_dim is None:
                env_dim = os.getenv("VERTEX_AI_EMBEDDING_DIM")
                if env_dim:
                    try:
                        resolved_dim = int(env_dim)
                    except ValueError:
                        print(f"[WARN] Ignoring invalid VERTEX_AI_EMBEDDING_DIM value: {env_dim}", file=sys.stderr)
            embeddings = maybe_compute_embeddings_vertex_ai(
                sections,
                project=resolved_project,
                location=resolved_location,
                model_name=resolved_model,
                dimensionality=resolved_dim,
                normalize=embedding_normalize,
            )
        elif embedding_provider == "gemini":
            embeddings = maybe_compute_embeddings_gemini(
                sections,
                api_key=gemini_api_key,
                model_name=gemini_embedding_model or "models/text-embedding-004",
                task_type=gemini_task_type,
                normalize=embedding_normalize,
                batch_size=gemini_batch_size,
            )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
        if embeddings:
            record["embeddings"] = embeddings

    if classify_ccs:
        taxonomy_path = ccs_taxonomy_path
        if not taxonomy_path.exists():
            raise ValueError(f"CCS taxonomy XML not found: {taxonomy_path}")
        taxonomy = load_taxonomy(taxonomy_path)
        classifier = CCSClassifier(
            taxonomy,
            embedding_model=None if ccs_embedding_model in (None, "", "none") else ccs_embedding_model,
        )
        summary_context = summary_to_prompt_text(record)
        if not summary_context.strip():
            raise ValueError("Cannot classify CCS concepts because the summary is empty.")
        try:
            outcome = classifier.classify(
                record,
                client,
                model=ccs_model,
                max_concepts=ccs_max_concepts,
                top_candidates=ccs_top_candidates,
                fallback_candidates=ccs_fallback_candidates,
                temperature=ccs_temperature,
                max_output_tokens=ccs_max_output_tokens,
                summary_text=summary_context,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Failed to classify CCS concepts: {exc}") from exc
        record = outcome.record
        record.setdefault("ccs_meta", {})
        record["ccs_meta"]["taxonomy"] = str(taxonomy.source_path or taxonomy_path)
        record["ccs_meta"]["model"] = ccs_model
        record["ccs_meta"]["predictions"] = outcome.predictions
        record["ccs_meta"]["candidates"] = [
            {
                "id": candidate.concept.concept_id,
                "path": candidate.concept.full_path_string(),
                "score": candidate.score,
            }
            for candidate in outcome.candidates
        ]
        record["ccs_meta"]["prompt"] = outcome.prompt

    return record


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise a PDF with the OpenAI API and emit structured JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file.")
    parser.add_argument("-o", "--output", type=Path, help="Where to write the JSON output (defaults to stdout).")
    parser.add_argument("--env-file", type=Path, help="Optional .env file with OPENAI_API_KEY.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name to use.")
    parser.add_argument(
        "--extractor",
        choices=["pypdf", "grobid"],
        default="pypdf",
        help="PDF extraction backend to use (default: pypdf).",
    )
    parser.add_argument(
        "--grobid-url",
        help="Base URL for a running GROBID service (used when --extractor=grobid).",
    )
    parser.add_argument(
        "--grobid-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the GROBID health check (used when --extractor=grobid).",
    )
    parser.add_argument("--language", default="Japanese", help="Language for the summaries (e.g. 'Japanese', 'English').")
    parser.add_argument("--chunk-size", type=int, default=2500, help="Approximate number of characters per chunk.")
    parser.add_argument("--overlap", type=int, default=250, help="Overlap between chunks in characters.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for all LLM calls.")
    parser.add_argument("--chunk-max-output", type=int, default=400, help="Maximum tokens for each chunk summary call.")
    parser.add_argument("--final-max-output", type=int, default=900, help="Maximum tokens for the final synthesis call.")
    parser.add_argument("--metadata-chars", type=int, default=4000, help="Characters from the PDF start used for metadata extraction prompts.")

    parser.add_argument("--paper-id", help="Preferred identifier (e.g. DOI) for the output JSON.")
    parser.add_argument("--title", help="Override paper title used in the output JSON.")
    parser.add_argument("--author", dest="authors", action="append", help="Repeatable author name override.")
    parser.add_argument("--year", type=int, help="Override publication year.")
    parser.add_argument("--pdf-link", help="Override PDF link for the output JSON.")
    parser.add_argument("--code-link", help="Override code link for the output JSON.")
    parser.add_argument("--ccs-path", dest="ccs_paths", action="append", help="Add a CCS path classification.")
    parser.add_argument("--ccs-id", dest="ccs_ids", action="append", help="Add a CCS identifier.")

    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Compute embeddings for each summary section (positioning/purpose/method/evaluation and abstract when available).",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["local", "vertex-ai", "gemini"],
        default="local",
        help="Embedding backend to use when --embeddings is enabled.",
    )
    parser.add_argument(
        "--embedding-model",
        default="intfloat/multilingual-e5-large-instruct",
        help="SentenceTransformer model to use when --embedding-provider=local.",
    )
    parser.add_argument("--no-embedding-normalize", action="store_false", dest="embedding_normalize", help="Disable L2 normalisation of embeddings.")
    parser.add_argument(
        "--vertex-project",
        help="Google Cloud project ID for Vertex AI embeddings (defaults to VERTEX_AI_PROJECT or GOOGLE_CLOUD_PROJECT env vars).",
    )
    parser.add_argument(
        "--vertex-location",
        help="Vertex AI location/region (e.g. 'us-central1', defaults to VERTEX_AI_LOCATION or 'us-central1').",
    )
    parser.add_argument(
        "--vertex-embedding-model",
        help="Vertex AI text embedding model name (e.g. 'text-embedding-004', defaults to VERTEX_AI_EMBEDDING_MODEL or 'text-embedding-004').",
    )
    parser.add_argument(
        "--vertex-embedding-dim",
        type=int,
        help="Optional output dimensionality when using Vertex AI embeddings.",
    )
    parser.add_argument(
        "--gemini-embedding-model",
        default="models/text-embedding-004",
        help="Gemini embedding model name (used when --embedding-provider=gemini).",
    )
    parser.add_argument(
        "--gemini-task-type",
        default="SEMANTIC_SIMILARITY",
        help="Task type passed to the Gemini embedding API.",
    )
    parser.add_argument(
        "--gemini-api-key",
        help="Override GEMINI_API_KEY when using Gemini embeddings.",
    )
    parser.add_argument(
        "--gemini-batch-size",
        type=int,
        default=32,
        help="Maximum sections per Gemini embeddings request.",
    )
    parser.set_defaults(embedding_normalize=True)

    parser.add_argument(
        "--classify-ccs",
        action="store_true",
        help="Run ACM CCS classification on the generated summary.",
    )
    parser.add_argument(
        "--ccs-taxonomy",
        type=Path,
        default=REPO_ROOT / "ACM CCS" / "acm_ccs2012-1626988337597.xml",
        help="Path to the ACM CCS taxonomy XML file.",
    )
    parser.add_argument(
        "--ccs-model",
        default=os.getenv("CCS_CLASSIFIER_MODEL", "gpt-4.1-mini"),
        help="OpenAI model to use for CCS classification.",
    )
    parser.add_argument("--ccs-max-concepts", type=int, default=3, help="Maximum CCS concepts to assign.")
    parser.add_argument("--ccs-top-candidates", type=int, default=15, help="Candidate concepts to surface to the LLM.")
    parser.add_argument(
        "--ccs-fallback-candidates",
        type=int,
        default=25,
        help="Candidates to surface when embeddings are unavailable.",
    )
    parser.add_argument("--ccs-temperature", type=float, default=0.1, help="Sampling temperature for CCS classification.")
    parser.add_argument(
        "--ccs-max-output",
        type=int,
        default=600,
        help="Maximum tokens for the CCS classification response.",
    )
    parser.add_argument(
        "--ccs-embedding-model",
        default=os.getenv("CCS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        help="SentenceTransformer model used for CCS candidate retrieval (use 'none' to disable embeddings).",
    )

    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        env_path = load_env(explicit=args.env_file, start=Path.cwd())
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}", file=sys.stderr)
        env_path = None
    if env_path:
        print(f"[INFO] Loaded environment variables from {env_path}", file=sys.stderr)

    try:
        record = summarise_pdf(
            args.pdf,
            extractor=args.extractor,
            grobid_url=args.grobid_url,
            grobid_timeout=args.grobid_timeout,
            model=args.model,
            language=args.language,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            temperature=args.temperature,
            chunk_max_tokens=args.chunk_max_output,
            final_max_tokens=args.final_max_output,
            metadata_chars=args.metadata_chars,
            paper_id=args.paper_id,
            title=args.title,
            authors=args.authors,
            year=args.year,
            pdf_link=args.pdf_link,
            code_link=args.code_link,
            ccs_paths_cli=args.ccs_paths,
            ccs_ids_cli=args.ccs_ids,
            compute_embeddings=args.embeddings,
            embedding_model=args.embedding_model,
            embedding_normalize=args.embedding_normalize,
            embedding_provider=args.embedding_provider,
            vertex_project=args.vertex_project,
            vertex_location=args.vertex_location,
            vertex_embedding_model=args.vertex_embedding_model,
            vertex_embedding_dim=args.vertex_embedding_dim,
            gemini_api_key=args.gemini_api_key,
            gemini_embedding_model=args.gemini_embedding_model,
            gemini_task_type=args.gemini_task_type,
            gemini_batch_size=args.gemini_batch_size,
            classify_ccs=args.classify_ccs,
            ccs_taxonomy_path=args.ccs_taxonomy,
            ccs_model=args.ccs_model,
            ccs_max_concepts=args.ccs_max_concepts,
            ccs_top_candidates=args.ccs_top_candidates,
            ccs_fallback_candidates=args.ccs_fallback_candidates,
            ccs_temperature=args.ccs_temperature,
            ccs_max_output_tokens=args.ccs_max_output,
            ccs_embedding_model=args.ccs_embedding_model,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    output_json = json.dumps(
        record,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json, encoding="utf-8")
        print(f"[INFO] Wrote summary to {args.output}", file=sys.stderr)
    else:
        print(output_json)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
