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
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

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


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        os.environ.setdefault(key, value)


def find_env_file(start_dir: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit:
        explicit = explicit.expanduser()
        if explicit.exists():
            return explicit
        print(f"[WARN] .env file not found: {explicit}", file=sys.stderr)
        return None

    for parent in [start_dir, *start_dir.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None


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


def read_pdf_text(path: Path) -> str:
    pdf_path = path.expanduser()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "Neither \"pypdf\" nor \"PyPDF2\" is available. Install one with \"pip install pypdf\"."
            ) from exc

    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
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
        raise ValueError("No extractable text found in the PDF.")
    return "\n\n".join(pages)


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
    heading_lines = "\n".join(
        f"- {label}:" for label in (headings["overview"], headings["positioning"], headings["solution"], headings["evaluation"])
    )
    user_prompt = (
        f"You will get a portion of a research paper. Using only that portion, "
        f"write concise bullet notes in {language}. Group the notes under the headings listed below. "
        f"Each heading should have 1-3 bullets. If the excerpt lacks information for a heading, write '{missing_token}'."
        "\n\nHeadings:\n"
        f"{heading_lines}\n\n"
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


def maybe_compute_embeddings(
    sections: Dict[str, str],
    *,
    model_name: str,
    normalize: bool,
) -> Dict[str, object]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np  # type: ignore  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "Embedding generation requires \"sentence-transformers\". Install it with \"pip install sentence-transformers\"."
        ) from exc

    texts: List[str] = []
    keys: List[str] = []
    for key in ("purpose", "method", "evaluation"):
        text = sections.get(key)
        if text and text != "":
            keys.append(key)
            texts.append(text)
    if not texts:
        return {}

    embedder = SentenceTransformer(model_name)
    vectors = embedder.encode(texts, normalize_embeddings=normalize)

    result: Dict[str, object] = {}
    for key, vector in zip(keys, vectors):
        result[key] = [float(x) for x in vector]
    dimension = len(vectors[0])
    result["model"] = model_name
    result["dim"] = dimension
    result["normed"] = normalize
    return result


def prefer_cli(cli_value, extracted_value):
    return cli_value if cli_value not in (None, "", [], {}) else extracted_value


def summarise_pdf(
    pdf_path: Path,
    *,
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
) -> Dict[str, object]:
    text = read_pdf_text(pdf_path)
    client = load_openai_client()

    excerpt = text[:metadata_chars]
    metadata = extract_metadata(
        client,
        model=model,
        excerpt=excerpt,
        temperature=max(0.0, min(temperature, 0.5)),
        max_tokens=400,
    )

    headings = section_labels_for(language)
    missing_token = missing_text_token(language)

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
        "title_hint": title or metadata.get("title"),
        "authors_hint": authors or metadata.get("authors"),
        "doi": metadata.get("doi"),
        "year_hint": year or metadata.get("year"),
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
    record["id"] = paper_id or structured.get("doi") or metadata.get("doi")
    record["title"] = prefer_cli(title, structured.get("title") or metadata.get("title"))

    authors_value = prefer_cli(authors, structured.get("authors") or metadata.get("authors"))
    if isinstance(authors_value, str):
        authors_list = [a.strip() for a in authors_value.split(",") if a.strip()]
    else:
        authors_list = list(authors_value or [])
    record["authors"] = authors_list

    record["abstract"] = structured.get("abstract") or metadata.get("abstract") or missing_token
    record["positioning_summary"] = structured.get("positioning_summary") or missing_token
    record["purpose_summary"] = structured.get("purpose_summary") or missing_token
    record["method_summary"] = structured.get("method_summary") or missing_token
    record["evaluation_summary"] = structured.get("evaluation_summary") or missing_token

    chosen_year = prefer_cli(year, structured.get("year") or metadata.get("year"))
    record["year"] = int(chosen_year) if chosen_year else None

    ccs_paths = prefer_cli(ccs_paths_cli, structured.get("ccs_paths")) or []
    ccs_ids = prefer_cli(ccs_ids_cli, structured.get("ccs_ids")) or []
    record["ccs"] = {"paths": list(ccs_paths), "ids": list(ccs_ids)}

    record["links"] = {
        "pdf": pdf_link or structured.get("pdf") or metadata.get("pdf"),
        "code": code_link or structured.get("code") or metadata.get("code"),
    }

    if compute_embeddings:
        sections = {
            "purpose": record.get("purpose_summary"),
            "method": record.get("method_summary"),
            "evaluation": record.get("evaluation_summary"),
        }
        embeddings = maybe_compute_embeddings(
            sections,
            model_name=embedding_model,
            normalize=embedding_normalize,
        )
        if embeddings:
            record["embeddings"] = embeddings

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
    parser.add_argument("--language", default="Japanese", help="Language for the summaries (e.g. 'Japanese', 'English').")
    parser.add_argument("--chunk-size", type=int, default=6000, help="Approximate number of characters per chunk.")
    parser.add_argument("--overlap", type=int, default=500, help="Overlap between chunks in characters.")
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

    parser.add_argument("--embeddings", action="store_true", help="Compute sentence-transformer embeddings for purpose/method/evaluation summaries.")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-large-instruct", help="SentenceTransformer model for embeddings.")
    parser.add_argument("--no-embedding-normalize", action="store_false", dest="embedding_normalize", help="Disable L2 normalisation of embeddings.")
    parser.set_defaults(embedding_normalize=True)

    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    env_path = find_env_file(Path.cwd(), args.env_file)
    if env_path:
        load_env_file(env_path)
        print(f"[INFO] Loaded environment variables from {env_path}", file=sys.stderr)

    try:
        record = summarise_pdf(
            args.pdf,
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
