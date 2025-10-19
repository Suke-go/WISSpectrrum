#!/usr/bin/env python3
"""Streamlit frontend for the WISS preprocessing pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

# Ensure the preprocessing package is importable even when launched via `streamlit run`
_THIS_FILE = Path(__file__).resolve()
_UI_DIR = _THIS_FILE.parent
_PREPROCESS_ROOT = _UI_DIR.parent
for _candidate in (_PREPROCESS_ROOT, _UI_DIR):
    _path_str = str(_candidate)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import streamlit as st

from shared import (
    CommandRequest,
    DEFAULT_TAXONOMY,
    EmbeddingToolOptions,
    PipelineFormState,
    SECTION_CHOICES,
    prepare_embeddings_command,
    prepare_pipeline_command,
)
from utils.paths import STATE_DB_PATH, resolve_data_roots

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

_VENV_DIR = REPO_ROOT / ".venv"
if sys.platform == "win32":
    _VENV_PYTHON = _VENV_DIR / "Scripts" / "python.exe"
else:
    _VENV_PYTHON = _VENV_DIR / "bin" / "python"
PYTHON_EXECUTABLE = str(_VENV_PYTHON if _VENV_PYTHON.exists() else Path(sys.executable))

try:
    from utils.env import load_env  # type: ignore
except Exception:  # pragma: no cover
    def load_env(*_, **__) -> Optional[Path]:
        return None


def _value_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _split_lines(value: str) -> List[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def run_command(
    request: CommandRequest,
    placeholder: "st.delta_generator.DeltaGenerator",
    initial_lines: Optional[Iterable[str]] = None,
) -> List[str]:
    logs: List[str] = []

    def emit(message: str) -> None:
        logs.append(message)
        placeholder.code("\n".join(logs), language="text")

    for line in initial_lines or []:
        emit(line)

    emit(f"[RUN] {' '.join(request.command)}")
    try:
        process = subprocess.Popen(
            request.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(request.cwd),
            env=request.env,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            emit(raw_line.rstrip())
        return_code = process.wait()
        emit(f"[DONE] return code={return_code}")
    except Exception as exc:  # pragma: no cover - UI feedback path
        emit(f"[ERROR] {exc}")
    finally:
        if request.cleanup_paths:
            for path in request.cleanup_paths:
                if path and path.exists():
                    try:
                        path.unlink()
                        emit(f"[INFO] Removed temp file: {path}")
                    except OSError:
                        emit(f"[WARN] Failed to remove temp file: {path}")
    return logs


def ensure_defaults(defaults: dict) -> None:
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value


st.set_page_config(page_title="WISS Preprocessing Control Panel", layout="wide")

ensure_defaults(
    {
        "env_path": "",
        "openai_key": os.getenv("OPENAI_API_KEY", ""),
        "gemini_key": os.getenv("GEMINI_API_KEY", ""),
        "pipeline_log": [],
        "embedding_log": [],
    }
)

# migrate legacy session defaults to updated token/model settings
if st.session_state.get("pipeline_chunk_max_tokens") == "400":
    st.session_state["pipeline_chunk_max_tokens"] = "2200"
if st.session_state.get("pipeline_final_max_tokens") == "1200":
    st.session_state["pipeline_final_max_tokens"] = "1600"
if st.session_state.get("pipeline_ccs_embedding_model") == "sentence-transformers/all-MiniLM-L6-v2":
    st.session_state["pipeline_ccs_embedding_model"] = "gemini-embedding-001"
if st.session_state.get("pipeline_gemini_model") == "models/gemini-embedding-001":
    st.session_state["pipeline_gemini_model"] = "gemini-embedding-001"

PIPELINE_DEFAULTS = {
    "pipeline_pdf_dir": "",
    "pipeline_additional_dirs": "",
    "pipeline_data_root": os.getenv("WISS_DATA_ROOT", ", ".join(str(path) for path in resolve_data_roots())),
    "pipeline_db_path": os.getenv("WISS_PIPELINE_DB", str(STATE_DB_PATH)),
    "pipeline_output_dir": str(REPO_ROOT / "output" / "summaries"),
    "pipeline_pattern": "*.pdf",
    "pipeline_limit": "",
    "pipeline_workers": "1",
    "pipeline_json_policy": "skip",
    "pipeline_summary_model": "gpt-5-mini",
    "pipeline_summary_language": "Japanese",
    "pipeline_chunk_size": "2500",
    "pipeline_overlap": "250",
    "pipeline_temperature": "0.2",
    "pipeline_chunk_max_tokens": "2200",
    "pipeline_final_max_tokens": "1600",
    "pipeline_metadata_chars": "4000",
    "pipeline_dual_language": False,
    "pipeline_extra_prompt": "",
    "pipeline_extractor": "pypdf",
    "pipeline_grobid_url": "http://localhost:8070",
    "pipeline_grobid_timeout": "60.0",
    "pipeline_compute_embeddings": True,
    "pipeline_embedding_provider": "gemini",
    "pipeline_local_model": "",
    "pipeline_vertex_project": "",
    "pipeline_vertex_location": "",
    "pipeline_vertex_model": "",
    "pipeline_vertex_dim": "",
    "pipeline_gemini_model": "gemini-embedding-001",
    "pipeline_gemini_task_type": "SEMANTIC_SIMILARITY",
    "pipeline_gemini_batch_size": "32",
    "pipeline_embedding_normalize": True,
    "pipeline_embedding_sections": list(SECTION_CHOICES),
    "pipeline_embedding_version": "",
    "pipeline_classify_ccs": True,
    "pipeline_ccs_model": "gpt-5-mini",
    "pipeline_ccs_taxonomy": str(DEFAULT_TAXONOMY),
    "pipeline_ccs_max_concepts": "3",
    "pipeline_ccs_top_candidates": "15",
    "pipeline_ccs_fallback_candidates": "25",
    "pipeline_ccs_temperature": "0.1",
    "pipeline_ccs_max_output_tokens": "900",
    "pipeline_ccs_embedding_model": "gemini-embedding-001",
}
ensure_defaults(PIPELINE_DEFAULTS)

EMBED_DEFAULTS = {
    "emb_summary_dir": str(REPO_ROOT / "output" / "summaries"),
    "emb_output_dir": "",
    "emb_provider": st.session_state["pipeline_embedding_provider"],
    "emb_normalize": st.session_state["pipeline_embedding_normalize"],
    "emb_sections": list(SECTION_CHOICES),
    "emb_embedding_version": st.session_state["pipeline_embedding_version"],
    "emb_force": False,
    "emb_dry_run": False,
    "emb_local_model": st.session_state["pipeline_local_model"],
    "emb_vertex_project": st.session_state["pipeline_vertex_project"],
    "emb_vertex_location": st.session_state["pipeline_vertex_location"],
    "emb_vertex_model": st.session_state["pipeline_vertex_model"],
    "emb_vertex_dim": st.session_state["pipeline_vertex_dim"],
    "emb_gemini_model": st.session_state["pipeline_gemini_model"],
    "emb_gemini_task_type": st.session_state["pipeline_gemini_task_type"],
    "emb_gemini_batch_size": st.session_state["pipeline_gemini_batch_size"],
}
ensure_defaults(EMBED_DEFAULTS)

st.title("WISS Preprocessing Control Panel (Streamlit)")

with st.sidebar:
    st.header("環境設定")
    env_path = st.text_input(".env パス", key="env_path")
    if st.button("環境変数を読み込み", use_container_width=True):
        try:
            loaded = load_env(explicit=Path(env_path)) if env_path else load_env()
            st.success(f"環境変数をロードしました: {loaded or '(default search)'}")
            current_openai = os.getenv("OPENAI_API_KEY")
            current_gemini = os.getenv("GEMINI_API_KEY")
            if current_openai and not st.session_state["openai_key"]:
                st.session_state["openai_key"] = current_openai
            if current_gemini and not st.session_state["gemini_key"]:
                st.session_state["gemini_key"] = current_gemini
        except FileNotFoundError as exc:
            st.error(str(exc))

    st.header("API Keys")
    st.text_input("OpenAI API Key", type="password", key="openai_key")
    st.text_input("Gemini API Key", type="password", key="gemini_key")

tabs = st.tabs(["パイプライン", "Embedding 再計算"])

with tabs[0]:
    pipeline_log_placeholder = st.empty()
    with st.form("pipeline_form"):
        st.subheader("入力ソース")
        col_left, col_right = st.columns(2)
        pdf_dir = col_left.text_input("PDF ディレクトリ", key="pipeline_pdf_dir")
        additional_dirs_raw = col_left.text_area("追加PDFディレクトリ (改行区切り)", key="pipeline_additional_dirs", height=110)
        data_root = col_left.text_input("基準ディレクトリ (WISS_DATA_ROOT, カンマ区切り)", key="pipeline_data_root")
        output_dir = col_right.text_input("出力ディレクトリ", key="pipeline_output_dir")
        db_path = col_right.text_input("ジョブDBパス (--db / WISS_PIPELINE_DB)", key="pipeline_db_path")
        limit = col_right.text_input("処理件数上限", key="pipeline_limit")
        workers = col_right.text_input("並列ワーカー数", key="pipeline_workers")
        pattern = col_right.text_input("ファイルパターン", key="pipeline_pattern")
        json_policy = col_right.selectbox(
            "既存 JSON の扱い",
            options=["skip", "force", "new"],
            format_func=lambda opt: {
                "skip": "既存があればスキップ",
                "force": "常に再処理 (--force)",
                "new": "新規のみ追加",
            }.get(opt, opt),
            key="pipeline_json_policy",
        )

        st.subheader("サマリー設定")
        col_s1, col_s2, col_s3 = st.columns(3)
        summary_model = col_s1.text_input("OpenAI モデル", key="pipeline_summary_model")
        summary_language = col_s2.text_input("要約言語", key="pipeline_summary_language")
        dual_language = col_s3.checkbox("英語要約も生成", key="pipeline_dual_language")

        col_s4, col_s5, col_s6 = st.columns(3)
        chunk_size = col_s4.text_input("chunk_size", key="pipeline_chunk_size")
        overlap = col_s5.text_input("overlap", key="pipeline_overlap")
        temperature = col_s6.text_input("温度", key="pipeline_temperature")

        col_s7, col_s8, col_s9 = st.columns(3)
        chunk_tokens = col_s7.text_input("chunk トークン上限", key="pipeline_chunk_max_tokens")
        final_tokens = col_s8.text_input("最終トークン上限", key="pipeline_final_max_tokens")
        metadata_chars = col_s9.text_input("メタデータ抽出文字数", key="pipeline_metadata_chars")

        extra_prompt = st.text_area("追加プロンプト", key="pipeline_extra_prompt", height=120)

        st.subheader("PDF 抽出設定")
        extractor = st.radio("抽出エンジン", options=["pypdf", "grobid"], key="pipeline_extractor", horizontal=True)
        grobid_cols = st.columns(2)
        grobid_url = grobid_cols[0].text_input("GROBID URL", key="pipeline_grobid_url", disabled=extractor != "grobid")
        grobid_timeout = grobid_cols[1].text_input("Timeout (秒)", key="pipeline_grobid_timeout", disabled=extractor != "grobid")

        st.subheader("Embedding 設定")
        compute_embeddings = st.checkbox("Embedding を計算する", key="pipeline_compute_embeddings")
        provider = st.selectbox("プロバイダ", options=["local", "vertex-ai", "gemini"], key="pipeline_embedding_provider")
        embedding_normalize = st.checkbox("ベクトルを正規化する", key="pipeline_embedding_normalize")
        embedding_sections = st.multiselect(
            "対象セクション (全選択で全体)",
            options=list(SECTION_CHOICES),
            key="pipeline_embedding_sections",
        )
        embedding_version = st.text_input("Embedding バージョン", key="pipeline_embedding_version")

        if provider == "local":
            st.text_input("ローカルモデル", key="pipeline_local_model")
        elif provider == "vertex-ai":
            col_v1, col_v2 = st.columns(2)
            col_v1.text_input("Vertex Project", key="pipeline_vertex_project")
            col_v1.text_input("Vertex Location", key="pipeline_vertex_location")
            col_v2.text_input("Embedding Model", key="pipeline_vertex_model")
            col_v2.text_input("出力次元", key="pipeline_vertex_dim")
        elif provider == "gemini":
            col_g1, col_g2, col_g3 = st.columns(3)
            col_g1.text_input("Gemini モデル", key="pipeline_gemini_model")
            col_g2.text_input("Gemini Task Type", key="pipeline_gemini_task_type")
            col_g3.text_input("Gemini Batch Size", key="pipeline_gemini_batch_size")

        st.subheader("ACM CCS 分類")
        classify_ccs = st.checkbox("CCS 分類を実行する", key="pipeline_classify_ccs")
        if classify_ccs:
            col_c1, col_c2 = st.columns(2)
            col_c1.text_input("CCS モデル", key="pipeline_ccs_model")
            col_c1.text_input("最大概念数", key="pipeline_ccs_max_concepts")
            col_c1.text_input("Top 候補数", key="pipeline_ccs_top_candidates")
            col_c2.text_input("Fallback 候補数", key="pipeline_ccs_fallback_candidates")
            col_c2.text_input("温度", key="pipeline_ccs_temperature")
            col_c2.text_input("最大出力トークン", key="pipeline_ccs_max_output_tokens")
            st.text_input("Embedding モデル", key="pipeline_ccs_embedding_model")
            st.text_input("タクソノミー XML パス", key="pipeline_ccs_taxonomy")

        submitted_pipeline = st.form_submit_button("パイプラインを実行する")

    if submitted_pipeline:
        pdf_dir_clean = pdf_dir.strip()
        if not pdf_dir_clean:
            st.error("PDF ディレクトリを入力してください。")
        else:
            additional_pdf_dirs = _split_lines(additional_dirs_raw)
            form = PipelineFormState(
                pdf_dirs=[pdf_dir_clean, *additional_pdf_dirs],
                data_roots=_value_or_none(data_root),
                db_path=_value_or_none(db_path),
                output_dir=_value_or_none(output_dir),
                pattern=_value_or_none(pattern),
                limit=_value_or_none(limit),
                workers=_value_or_none(workers),
                json_policy=json_policy,
                env_file=_value_or_none(st.session_state["env_path"]),
                summary_model=_value_or_none(summary_model),
                summary_language=_value_or_none(summary_language),
                dual_language=dual_language,
                chunk_size=_value_or_none(chunk_size),
                overlap=_value_or_none(overlap),
                temperature=_value_or_none(temperature),
                chunk_max_tokens=_value_or_none(chunk_tokens),
                final_max_tokens=_value_or_none(final_tokens),
                metadata_chars=_value_or_none(metadata_chars),
                extra_prompt=_value_or_none(extra_prompt),
                extractor=extractor,
                grobid_url=_value_or_none(grobid_url) if extractor == "grobid" else None,
                grobid_timeout=_value_or_none(grobid_timeout) if extractor == "grobid" else None,
                compute_embeddings=compute_embeddings,
                embedding_provider=_value_or_none(provider),
                embedding_model_local=_value_or_none(st.session_state["pipeline_local_model"]),
                vertex_project=_value_or_none(st.session_state["pipeline_vertex_project"]),
                vertex_location=_value_or_none(st.session_state["pipeline_vertex_location"]),
                vertex_model=_value_or_none(st.session_state["pipeline_vertex_model"]),
                vertex_dim=_value_or_none(st.session_state["pipeline_vertex_dim"]),
                gemini_model=_value_or_none(st.session_state["pipeline_gemini_model"]),
                gemini_task_type=_value_or_none(st.session_state["pipeline_gemini_task_type"]),
                gemini_batch_size=_value_or_none(st.session_state["pipeline_gemini_batch_size"]),
                embedding_normalize=embedding_normalize,
                embedding_sections=embedding_sections,
                embedding_version=_value_or_none(embedding_version),
                classify_ccs=classify_ccs,
                ccs_model=_value_or_none(st.session_state["pipeline_ccs_model"]),
                ccs_taxonomy=_value_or_none(st.session_state["pipeline_ccs_taxonomy"]),
                ccs_max_concepts=_value_or_none(st.session_state["pipeline_ccs_max_concepts"]),
                ccs_top_candidates=_value_or_none(st.session_state["pipeline_ccs_top_candidates"]),
                ccs_fallback_candidates=_value_or_none(st.session_state["pipeline_ccs_fallback_candidates"]),
                ccs_temperature=_value_or_none(st.session_state["pipeline_ccs_temperature"]),
                ccs_max_output_tokens=_value_or_none(st.session_state["pipeline_ccs_max_output_tokens"]),
                ccs_embedding_model=_value_or_none(st.session_state["pipeline_ccs_embedding_model"]),
                openai_key=_value_or_none(st.session_state["openai_key"]),
                gemini_api_key=_value_or_none(st.session_state["gemini_key"]),
            )
            try:
                request, config_path = prepare_pipeline_command(form, PYTHON_EXECUTABLE, REPO_ROOT)
            except ValueError as exc:
                st.error(str(exc))
            else:
                initial_lines = [f"[INFO] Config written: {config_path}"] if config_path else None
                st.session_state["pipeline_log"] = run_command(request, pipeline_log_placeholder, initial_lines)

    if st.session_state["pipeline_log"]:
        pipeline_log_placeholder.code("\n".join(st.session_state["pipeline_log"]), language="text")

with tabs[1]:
    embedding_log_placeholder = st.empty()
    with st.form("embedding_form"):
        st.subheader("サマリー JSON ルート")
        summary_dir = st.text_input("ディレクトリ", key="emb_summary_dir")
        output_dir = st.text_input("出力ディレクトリ (省略可)", key="emb_output_dir")
        provider = st.selectbox("プロバイダ", options=["local", "vertex-ai", "gemini"], key="emb_provider")
        normalize = st.checkbox("ベクトルを正規化する", key="emb_normalize")
        sections = st.multiselect(
            "対象セクション (全選択で全体)",
            options=list(SECTION_CHOICES),
            key="emb_sections",
        )
        embedding_version = st.text_input("Embedding バージョン", key="emb_embedding_version")
        force = st.checkbox("--force を付与", key="emb_force")
        dry_run = st.checkbox("--dry-run を付与", key="emb_dry_run")

        if provider == "local":
            st.text_input("ローカルモデル", key="emb_local_model")
        elif provider == "vertex-ai":
            col_ev1, col_ev2 = st.columns(2)
            col_ev1.text_input("Vertex Project", key="emb_vertex_project")
            col_ev1.text_input("Vertex Location", key="emb_vertex_location")
            col_ev2.text_input("Embedding Model", key="emb_vertex_model")
            col_ev2.text_input("出力次元", key="emb_vertex_dim")
        elif provider == "gemini":
            col_eg1, col_eg2, col_eg3 = st.columns(3)
            col_eg1.text_input("Gemini モデル", key="emb_gemini_model")
            col_eg2.text_input("Gemini Task Type", key="emb_gemini_task_type")
            col_eg3.text_input("Gemini Batch Size", key="emb_gemini_batch_size")

        submitted_emb = st.form_submit_button("Embedding を再計算する")

    if submitted_emb:
        summary_dir_clean = summary_dir.strip()
        if not summary_dir_clean:
            st.error("サマリー JSON のディレクトリを入力してください。")
        else:
            options = EmbeddingToolOptions(
                summary_dir=summary_dir_clean,
                provider=_value_or_none(provider),
                normalize=normalize,
                sections=sections,
                embedding_version=_value_or_none(embedding_version),
                force=force,
                dry_run=dry_run,
                output_dir=_value_or_none(output_dir),
                env_file=_value_or_none(st.session_state["env_path"]),
                local_model=_value_or_none(st.session_state["emb_local_model"]),
                vertex_project=_value_or_none(st.session_state["emb_vertex_project"]),
                vertex_location=_value_or_none(st.session_state["emb_vertex_location"]),
                vertex_model=_value_or_none(st.session_state["emb_vertex_model"]),
                vertex_dim=_value_or_none(st.session_state["emb_vertex_dim"]),
                gemini_model=_value_or_none(st.session_state["emb_gemini_model"]),
                gemini_api_key=_value_or_none(st.session_state["gemini_key"]),
                gemini_task_type=_value_or_none(st.session_state["emb_gemini_task_type"]),
                gemini_batch_size=_value_or_none(st.session_state["emb_gemini_batch_size"]),
                openai_key=_value_or_none(st.session_state["openai_key"]),
            )
            request = prepare_embeddings_command(options, PYTHON_EXECUTABLE, REPO_ROOT)
            st.session_state["embedding_log"] = run_command(request, embedding_log_placeholder)

    if st.session_state["embedding_log"]:
        embedding_log_placeholder.code("\n".join(st.session_state["embedding_log"]), language="text")
