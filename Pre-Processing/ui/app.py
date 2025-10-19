#!/usr/bin/env python3
"""ローカル前処理パイプラインを操作する Tkinter ベースの簡易フロントエンド。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from typing import Dict, List, Optional

from shared import (
    CommandRequest,
    DEFAULT_TAXONOMY,
    EmbeddingToolOptions,
    PipelineFormState,
    SECTION_CHOICES,
    prepare_embeddings_command,
    prepare_pipeline_command,
)

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_VENV_DIR = REPO_ROOT / ".venv"
if sys.platform == "win32":
    _VENV_PYTHON = _VENV_DIR / "Scripts" / "python.exe"
else:
    _VENV_PYTHON = _VENV_DIR / "bin" / "python"
PYTHON_EXECUTABLE = _VENV_PYTHON if _VENV_PYTHON.exists() else Path(sys.executable)

try:
    from utils.env import load_env  # type: ignore
except Exception:  # pragma: no cover - fallback when running in isolation
    def load_env(*_, **__) -> Optional[Path]:
        return None


class PreprocessingUI:
    """Tkinter-based control panel for orchestrating preprocessing tasks."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("WISS Preprocessing Control Panel")
        self.root.geometry("1180x860")
        self.root.minsize(1024, 720)

        self._is_running = False
        self._active_thread: Optional[threading.Thread] = None

        self._init_variables()
        self._build_layout()

        self.python_exec = str(PYTHON_EXECUTABLE)

    # ------------------------------------------------------------------ UI setup
    def _init_variables(self) -> None:
        self.pdf_dir_var = tk.StringVar()
        self.additional_pdf_dirs: List[str] = []
        self.output_dir_var = tk.StringVar(value=str(REPO_ROOT / "output" / "summaries"))
        self.env_path_var = tk.StringVar()
        self.pattern_var = tk.StringVar(value="*.pdf")
        self.limit_var = tk.StringVar()
        self.json_policy_var = tk.StringVar(value="skip")

        self.summary_model_var = tk.StringVar(value="gpt-5-mini")
        self.summary_language_var = tk.StringVar(value="Japanese")
        self.chunk_size_var = tk.StringVar(value="2500")
        self.overlap_var = tk.StringVar(value="250")
        self.temperature_var = tk.StringVar(value="0.2")
        self.chunk_max_tokens_var = tk.StringVar(value="2200")
        self.final_max_tokens_var = tk.StringVar(value="1600")
        self.metadata_chars_var = tk.StringVar(value="4000")
        self.dual_language_var = tk.BooleanVar(value=False)
        self.extractor_var = tk.StringVar(value="pypdf")
        self.grobid_url_var = tk.StringVar(value="http://localhost:8070")
        self.grobid_timeout_var = tk.StringVar(value="60.0")
        self.grobid_status_var = tk.StringVar(value="未確認")
        self.compute_embeddings_var = tk.BooleanVar(value=True)
        self.embedding_provider_var = tk.StringVar(value="gemini")
        self.embedding_model_var = tk.StringVar(value="intfloat/multilingual-e5-large-instruct")
        self.embedding_normalize_var = tk.BooleanVar(value=True)
        self.embedding_version_var = tk.StringVar(value="v1")
        self.embedding_force_var = tk.BooleanVar(value=False)
        self.embedding_dry_run_var = tk.BooleanVar(value=False)
        self.vertex_project_var = tk.StringVar()
        self.vertex_location_var = tk.StringVar(value="us-central1")
        self.vertex_model_var = tk.StringVar(value="text-embedding-004")
        self.vertex_dim_var = tk.StringVar()
        self.gemini_model_var = tk.StringVar(value="gemini-embedding-001")
        self.gemini_task_type_var = tk.StringVar(value="SEMANTIC_SIMILARITY")
        self.gemini_batch_size_var = tk.StringVar(value="32")
        self.gemini_api_key_var = tk.StringVar()
        self.local_model_var = tk.StringVar(value="intfloat/multilingual-e5-large-instruct")
        self.classify_ccs_var = tk.BooleanVar(value=True)
        self.ccs_model_var = tk.StringVar(value="gpt-5-mini")
        self.ccs_taxonomy_var = tk.StringVar(value=str(DEFAULT_TAXONOMY))
        self.ccs_max_concepts_var = tk.StringVar(value="3")
        self.ccs_top_candidates_var = tk.StringVar(value="15")
        self.ccs_fallback_candidates_var = tk.StringVar(value="25")
        self.ccs_temperature_var = tk.StringVar(value="0.1")
        self.ccs_max_output_tokens_var = tk.StringVar(value="900")
        self.ccs_embedding_model_var = tk.StringVar(value="gemini-embedding-001")

        self.embedding_sections_vars: Dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=True) for name in SECTION_CHOICES
        }

        self.openai_key_ui_var = tk.StringVar()
        self.model_search_var = tk.StringVar()

        self.emb_summary_dir_var = tk.StringVar(value=str(REPO_ROOT / "output" / "summaries"))
        self.emb_output_dir_var = tk.StringVar()

    def _build_layout(self) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        root.rowconfigure(1, weight=0)

        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        pipeline_frame = ttk.Frame(self.notebook)
        tools_frame = ttk.Frame(self.notebook)
        self.notebook.add(pipeline_frame, text="パイプライン")
        self.notebook.add(tools_frame, text="ツール / 設定")

        self._build_pipeline_tab(pipeline_frame)
        self._build_tools_tab(tools_frame)

        log_frame = ttk.LabelFrame(root, text="ログ")
        log_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = ScrolledText(log_frame, height=12, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")

    # ------------------------------------------------------------------ pipeline tab
    def _build_pipeline_tab(self, container: ttk.Frame) -> None:
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

        source_frame = ttk.LabelFrame(container, text="入力ソース")
        source_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=8, pady=6)
        source_frame.columnconfigure(1, weight=1)

        ttk.Label(source_frame, text="PDF ディレクトリ").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        pdf_entry = ttk.Entry(source_frame, textvariable=self.pdf_dir_var)
        pdf_entry.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(source_frame, text="選択", command=self._choose_pdf_dir).grid(row=0, column=2, padx=4, pady=4)
        ttk.Button(source_frame, text="追加", command=self._add_pdf_dir).grid(row=0, column=3, padx=4, pady=4)

        self.additional_dirs_label = ttk.Label(source_frame, text="追加ディレクトリ: なし")
        self.additional_dirs_label.grid(row=1, column=1, columnspan=3, sticky="w", padx=4, pady=2)

        ttk.Label(source_frame, text="出力ディレクトリ").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        output_entry = ttk.Entry(source_frame, textvariable=self.output_dir_var)
        output_entry.grid(row=2, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(source_frame, text="選択", command=self._choose_output_dir).grid(row=2, column=2, padx=4, pady=4)

        ttk.Label(source_frame, text="処理件数上限").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(source_frame, textvariable=self.limit_var, width=10).grid(row=3, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(source_frame, text="ファイルパターン").grid(row=3, column=2, sticky="e", padx=4, pady=4)
        ttk.Entry(source_frame, textvariable=self.pattern_var, width=16).grid(row=3, column=3, sticky="w", padx=4, pady=4)

        policy_frame = ttk.LabelFrame(container, text="既存 JSON の扱い")
        policy_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)
        ttk.Radiobutton(policy_frame, text="既存があればスキップ", value="skip", variable=self.json_policy_var).grid(row=0, column=0, sticky="w", padx=4, pady=3)
        ttk.Radiobutton(policy_frame, text="常に再処理して上書き (--force)", value="force", variable=self.json_policy_var).grid(row=1, column=0, sticky="w", padx=4, pady=3)
        ttk.Radiobutton(policy_frame, text="新規のみ追加 (既存ファイルは保持)", value="new", variable=self.json_policy_var).grid(row=2, column=0, sticky="w", padx=4, pady=3)

        env_frame = ttk.LabelFrame(container, text="環境変数")
        env_frame.grid(row=1, column=1, sticky="nsew", padx=8, pady=6)
        env_frame.columnconfigure(1, weight=1)
        ttk.Label(env_frame, text=".env パス").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(env_frame, textvariable=self.env_path_var).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(env_frame, text="選択", command=self._choose_env_path).grid(row=0, column=2, padx=4, pady=4)
        ttk.Button(env_frame, text="読み込み", command=self._load_env_clicked).grid(row=0, column=3, padx=4, pady=4)

        summary_frame = ttk.LabelFrame(container, text="サマリー設定")
        summary_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=6)
        summary_frame.columnconfigure(1, weight=1)
        ttk.Label(summary_frame, text="OpenAI モデル").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(summary_frame, textvariable=self.summary_model_var).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(summary_frame, text="モデル候補", command=self._open_models_tab).grid(row=0, column=2, padx=4, pady=4)

        ttk.Label(summary_frame, text="要約言語").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(summary_frame, textvariable=self.summary_language_var, width=16).grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(summary_frame, text="chunk_size / overlap").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        chunk_frame = ttk.Frame(summary_frame)
        chunk_frame.grid(row=2, column=1, sticky="w")
        ttk.Entry(chunk_frame, textvariable=self.chunk_size_var, width=8).grid(row=0, column=0, padx=(0, 4))
        ttk.Entry(chunk_frame, textvariable=self.overlap_var, width=8).grid(row=0, column=1, padx=(0, 4))

        ttk.Label(summary_frame, text="温度").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(summary_frame, textvariable=self.temperature_var, width=8).grid(row=3, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(summary_frame, text="トークン上限 (chunk / final)").grid(row=4, column=0, sticky="w", padx=4, pady=4)
        token_frame = ttk.Frame(summary_frame)
        token_frame.grid(row=4, column=1, sticky="w")
        ttk.Entry(token_frame, textvariable=self.chunk_max_tokens_var, width=10).grid(row=0, column=0, padx=(0, 4))
        ttk.Entry(token_frame, textvariable=self.final_max_tokens_var, width=10).grid(row=0, column=1, padx=(0, 4))

        ttk.Label(summary_frame, text="メタデータ抽出文字数").grid(row=5, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(summary_frame, textvariable=self.metadata_chars_var, width=10).grid(row=5, column=1, sticky="w", padx=4, pady=4)

        ttk.Checkbutton(
            summary_frame,
            text="英語要約も生成 (translations.en)",
            variable=self.dual_language_var,
        ).grid(row=6, column=0, columnspan=3, sticky="w", padx=4, pady=4)

        ttk.Label(summary_frame, text="追加プロンプト").grid(row=7, column=0, sticky="nw", padx=4, pady=4)
        self.summary_prompt_text = tk.Text(summary_frame, height=4, width=40)
        self.summary_prompt_text.grid(row=7, column=1, columnspan=2, sticky="ew", padx=4, pady=4)

        grobid_frame = ttk.LabelFrame(container, text="PDF 抽出設定")
        grobid_frame.grid(row=2, column=1, sticky="nsew", padx=8, pady=6)
        ttk.Radiobutton(grobid_frame, text="PyPDF (ローカル)", value="pypdf", variable=self.extractor_var, command=self._update_extractor_state).grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Radiobutton(grobid_frame, text="GROBID (Docker)", value="grobid", variable=self.extractor_var, command=self._update_extractor_state).grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(grobid_frame, text="GROBID URL").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.grobid_url_entry = ttk.Entry(grobid_frame, textvariable=self.grobid_url_var)
        self.grobid_url_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(grobid_frame, text="Timeout[s]").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        self.grobid_timeout_entry = ttk.Entry(grobid_frame, textvariable=self.grobid_timeout_var, width=8)
        self.grobid_timeout_entry.grid(row=2, column=1, sticky="w", padx=4, pady=4)

        status_frame = ttk.Frame(grobid_frame)
        status_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        ttk.Button(status_frame, text="状態チェック", command=self._check_grobid_status).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(status_frame, text="Docker 稼働確認", command=self._check_docker_for_grobid).grid(row=0, column=1, padx=(0, 6))
        ttk.Label(status_frame, textvariable=self.grobid_status_var).grid(row=0, column=2, sticky="w")

        embedding_frame = ttk.LabelFrame(container, text="Embedding 設定")
        embedding_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=6)
        embedding_frame.columnconfigure(1, weight=1)
        ttk.Checkbutton(embedding_frame, text="Embedding を計算する", variable=self.compute_embeddings_var, command=self._update_embedding_state).grid(row=0, column=0, columnspan=3, sticky="w", padx=4, pady=2)

        ttk.Label(embedding_frame, text="プロバイダ").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        provider_combo = ttk.Combobox(embedding_frame, textvariable=self.embedding_provider_var, values=["local", "vertex-ai", "gemini"], state="readonly", width=10)
        provider_combo.grid(row=1, column=1, sticky="w", padx=4, pady=4)
        provider_combo.bind("<<ComboboxSelected>>", lambda *_: self._update_embedding_state())

        self.embedding_stack = ttk.Frame(embedding_frame)
        self.embedding_stack.grid(row=2, column=0, columnspan=3, sticky="ew", padx=4, pady=4)

        self.local_frame = self._build_local_embedding_frame(self.embedding_stack)
        self.vertex_frame = self._build_vertex_embedding_frame(self.embedding_stack)
        self.gemini_frame = self._build_gemini_embedding_frame(self.embedding_stack)

        section_frame = ttk.Frame(embedding_frame)
        section_frame.grid(row=3, column=0, columnspan=3, sticky="w", padx=4, pady=4)
        ttk.Label(section_frame, text="対象セクション:").grid(row=0, column=0, sticky="w")
        for idx, name in enumerate(SECTION_CHOICES):
            ttk.Checkbutton(section_frame, text=name, variable=self.embedding_sections_vars[name]).grid(row=0, column=idx + 1, sticky="w", padx=2)

        ttk.Label(embedding_frame, text="バージョンタグ").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(embedding_frame, textvariable=self.embedding_version_var, width=12).grid(row=4, column=1, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(embedding_frame, text="L2 正規化 (--normalize)", variable=self.embedding_normalize_var).grid(row=5, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(embedding_frame, text="既存を強制再計算 (--force)", variable=self.embedding_force_var).grid(row=6, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(embedding_frame, text="Dry-run", variable=self.embedding_dry_run_var).grid(row=7, column=0, columnspan=2, sticky="w", padx=4, pady=2)

        ccs_frame = ttk.LabelFrame(container, text="CCS 分類設定")
        ccs_frame.grid(row=3, column=1, sticky="nsew", padx=8, pady=6)
        ccs_frame.columnconfigure(1, weight=1)
        ttk.Checkbutton(ccs_frame, text="CCS 分類を実行する", variable=self.classify_ccs_var).grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        ttk.Label(ccs_frame, text="モデル").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(ccs_frame, textvariable=self.ccs_model_var).grid(row=1, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(ccs_frame, text="taxonomy パス").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        taxonomy_entry = ttk.Entry(ccs_frame, textvariable=self.ccs_taxonomy_var)
        taxonomy_entry.grid(row=2, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(ccs_frame, text="選択", command=self._choose_taxonomy_path).grid(row=2, column=2, padx=4, pady=2)

        ttk.Label(ccs_frame, text="max concepts / top / fallback").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        ccs_counts = ttk.Frame(ccs_frame)
        ccs_counts.grid(row=3, column=1, sticky="w")
        ttk.Entry(ccs_counts, textvariable=self.ccs_max_concepts_var, width=6).grid(row=0, column=0, padx=(0, 4))
        ttk.Entry(ccs_counts, textvariable=self.ccs_top_candidates_var, width=6).grid(row=0, column=1, padx=(0, 4))
        ttk.Entry(ccs_counts, textvariable=self.ccs_fallback_candidates_var, width=6).grid(row=0, column=2, padx=(0, 4))

        ttk.Label(ccs_frame, text="温度 / トークン上限").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        ccs_temp = ttk.Frame(ccs_frame)
        ccs_temp.grid(row=4, column=1, sticky="w")
        ttk.Entry(ccs_temp, textvariable=self.ccs_temperature_var, width=6).grid(row=0, column=0, padx=(0, 4))
        ttk.Entry(ccs_temp, textvariable=self.ccs_max_output_tokens_var, width=6).grid(row=0, column=1, padx=(0, 4))

        ttk.Label(ccs_frame, text="埋め込みモデル").grid(row=5, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(ccs_frame, textvariable=self.ccs_embedding_model_var).grid(row=5, column=1, sticky="ew", padx=4, pady=2)

        buttons_frame = ttk.Frame(container)
        buttons_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=8, pady=10)
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        ttk.Button(buttons_frame, text="パイプライン実行", command=self._run_pipeline).grid(row=0, column=0, sticky="ew", padx=4)
        ttk.Button(buttons_frame, text="実行結果リセット", command=self._reset_log).grid(row=0, column=1, sticky="ew", padx=4)

        self._update_extractor_state()
        self._update_embedding_state()

    def _build_local_embedding_frame(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text="SentenceTransformer モデル").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(frame, textvariable=self.local_model_var, width=32).grid(row=0, column=1, sticky="w", padx=2, pady=2)
        return frame

    def _build_vertex_embedding_frame(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text="Project").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(frame, textvariable=self.vertex_project_var, width=18).grid(row=0, column=1, sticky="w", padx=2, pady=2)
        ttk.Label(frame, text="Location").grid(row=0, column=2, sticky="w", padx=2, pady=2)
        ttk.Entry(frame, textvariable=self.vertex_location_var, width=12).grid(row=0, column=3, sticky="w", padx=2, pady=2)
        ttk.Label(frame, text="Model").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(frame, textvariable=self.vertex_model_var, width=24).grid(row=1, column=1, sticky="w", padx=2, pady=2)
        ttk.Label(frame, text="出力次元").grid(row=1, column=2, sticky="w", padx=2, pady=2)
        ttk.Entry(frame, textvariable=self.vertex_dim_var, width=8).grid(row=1, column=3, sticky="w", padx=2, pady=2)
        return frame

    def _build_gemini_embedding_frame(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text="モデル").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(frame, textvariable=self.gemini_model_var, width=30).grid(row=0, column=1, sticky="w", padx=2, pady=2)
        ttk.Label(frame, text="タスク種別").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(frame, textvariable=self.gemini_task_type_var, width=20).grid(row=1, column=1, sticky="w", padx=2, pady=2)
        ttk.Label(frame, text="バッチサイズ").grid(row=1, column=2, sticky="w", padx=2, pady=2)
        ttk.Entry(frame, textvariable=self.gemini_batch_size_var, width=10).grid(row=1, column=3, sticky="w", padx=2, pady=2)
        ttk.Label(frame, text="API Key (任意)").grid(row=2, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(frame, textvariable=self.gemini_api_key_var, width=40, show="•").grid(row=2, column=1, columnspan=3, sticky="w", padx=2, pady=2)
        return frame

    # ------------------------------------------------------------------ tools tab
    def _build_tools_tab(self, container: ttk.Frame) -> None:
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

        models_frame = ttk.LabelFrame(container, text="OpenAI モデル検索")
        models_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=6)
        models_frame.columnconfigure(0, weight=1)
        ttk.Label(models_frame, text="API Key (未入力なら環境変数を使用)").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(models_frame, textvariable=self.openai_key_ui_var, show="•").grid(row=1, column=0, sticky="ew", padx=4, pady=2)
        search_row = ttk.Frame(models_frame)
        search_row.grid(row=2, column=0, sticky="ew", padx=4, pady=2)
        ttk.Entry(search_row, textvariable=self.model_search_var).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        search_row.columnconfigure(0, weight=1)
        ttk.Button(search_row, text="Fetch", command=self._fetch_models).grid(row=0, column=1, padx=(0, 4))
        ttk.Button(search_row, text="選択モデルを反映", command=self._apply_selected_model).grid(row=0, column=2)
        self.models_listbox = tk.Listbox(models_frame, height=15)
        self.models_listbox.grid(row=3, column=0, sticky="nsew", padx=4, pady=4)

        embedding_tool_frame = ttk.LabelFrame(container, text="Embedding 再計算ツール")
        embedding_tool_frame.grid(row=0, column=1, sticky="nsew", padx=8, pady=6)
        embedding_tool_frame.columnconfigure(1, weight=1)
        ttk.Label(embedding_tool_frame, text="サマリー JSON ルート").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(embedding_tool_frame, textvariable=self.emb_summary_dir_var).grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(embedding_tool_frame, text="選択", command=self._choose_emb_summary_dir).grid(row=0, column=2, padx=4, pady=2)

        ttk.Label(embedding_tool_frame, text="出力ディレクトリ (省略可)").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(embedding_tool_frame, textvariable=self.emb_output_dir_var).grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(embedding_tool_frame, text="選択", command=self._choose_emb_output_dir).grid(row=1, column=2, padx=4, pady=2)

        ttk.Button(embedding_tool_frame, text="Embedding を再計算", command=self._run_embeddings_only).grid(row=2, column=0, columnspan=3, sticky="ew", padx=4, pady=8)

    # ------------------------------------------------------------------ UI helpers
    def _choose_pdf_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.pdf_dir_var.get() or str(REPO_ROOT))
        if chosen:
            self.pdf_dir_var.set(chosen)

    def _add_pdf_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=str(REPO_ROOT))
        if chosen:
            path = str(Path(chosen).resolve())
            if path not in self.additional_pdf_dirs:
                self.additional_pdf_dirs.append(path)
            self._update_additional_dirs_label()

    def _update_additional_dirs_label(self) -> None:
        if not self.additional_pdf_dirs:
            text = "追加ディレクトリ: なし"
        else:
            text = f"追加ディレクトリ: {len(self.additional_pdf_dirs)} 件"
        self.additional_dirs_label.config(text=text)

    def _choose_output_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.output_dir_var.get() or str(REPO_ROOT))
        if chosen:
            self.output_dir_var.set(chosen)

    def _choose_env_path(self) -> None:
        chosen = filedialog.askopenfilename(initialdir=str(REPO_ROOT), filetypes=[("Env files", "*.env"), ("All files", "*.*")])
        if chosen:
            self.env_path_var.set(chosen)

    def _choose_taxonomy_path(self) -> None:
        chosen = filedialog.askopenfilename(initialdir=str(DEFAULT_TAXONOMY.parent), filetypes=[("XML", "*.xml"), ("All files", "*.*")])
        if chosen:
            self.ccs_taxonomy_var.set(chosen)

    def _choose_emb_summary_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.emb_summary_dir_var.get() or str(REPO_ROOT))
        if chosen:
            self.emb_summary_dir_var.set(chosen)

    def _choose_emb_output_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.emb_output_dir_var.get() or str(REPO_ROOT))
        if chosen:
            self.emb_output_dir_var.set(chosen)

    def _open_models_tab(self) -> None:
        self.notebook.select(1)

    def _reset_log(self) -> None:
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def _load_env_clicked(self) -> None:
        path = self.env_path_var.get().strip()
        try:
            loaded = load_env(explicit=Path(path)) if path else load_env()
        except FileNotFoundError as exc:
            messagebox.showerror("環境変数の読み込み失敗", str(exc))
            return
        self._append_log(f"[INFO] Loaded environment: {loaded or '(default search)'}")
        current_key = os.getenv("OPENAI_API_KEY")
        if current_key and not self.openai_key_ui_var.get():
            self.openai_key_ui_var.set(current_key)

    def _update_extractor_state(self) -> None:
        is_grobid = self.extractor_var.get() == "grobid"
        state = "normal" if is_grobid else "disabled"
        self.grobid_url_entry.config(state=state)
        self.grobid_timeout_entry.config(state=state)

    def _update_embedding_state(self) -> None:
        for child in self.embedding_stack.winfo_children():
            child.grid_remove()
        provider = self.embedding_provider_var.get()
        target_frame = self.local_frame
        if provider == "local":
            target_frame = self.local_frame
        elif provider == "vertex-ai":
            target_frame = self.vertex_frame
        elif provider == "gemini":
            target_frame = self.gemini_frame
        target_frame.grid(row=0, column=0, sticky="ew")

    # ------------------------------------------------------------------ pipeline execution
    def _run_pipeline(self) -> None:
        if self._is_running:
            messagebox.showwarning("実行中", "他の処理が実行中です。完了をお待ちください。")
            return

        pdf_root = self.pdf_dir_var.get().strip()
        if not pdf_root:
            messagebox.showerror("入力エラー", "PDF ディレクトリを指定してください。")
            return
        form = PipelineFormState(
            pdf_dirs=[pdf_root, *self.additional_pdf_dirs],
            output_dir=self._value_or_none(self.output_dir_var.get()),
            pattern=self._value_or_none(self.pattern_var.get()),
            limit=self._value_or_none(self.limit_var.get()),
            json_policy=self.json_policy_var.get(),
            env_file=self._value_or_none(self.env_path_var.get()),
            summary_model=self._value_or_none(self.summary_model_var.get()),
            summary_language=self._value_or_none(self.summary_language_var.get()),
            dual_language=self.dual_language_var.get(),
            chunk_size=self._value_or_none(self.chunk_size_var.get()),
            overlap=self._value_or_none(self.overlap_var.get()),
            temperature=self._value_or_none(self.temperature_var.get()),
            chunk_max_tokens=self._value_or_none(self.chunk_max_tokens_var.get()),
            final_max_tokens=self._value_or_none(self.final_max_tokens_var.get()),
            metadata_chars=self._value_or_none(self.metadata_chars_var.get()),
            extra_prompt=self._value_or_none(self.summary_prompt_text.get("1.0", tk.END)),
            extractor=self.extractor_var.get(),
            grobid_url=self._value_or_none(self.grobid_url_var.get()),
            grobid_timeout=self._value_or_none(self.grobid_timeout_var.get()),
            compute_embeddings=self.compute_embeddings_var.get(),
            embedding_provider=self._value_or_none(self.embedding_provider_var.get()),
            embedding_model_local=self._value_or_none(self.local_model_var.get()),
            vertex_project=self._value_or_none(self.vertex_project_var.get()),
            vertex_location=self._value_or_none(self.vertex_location_var.get()),
            vertex_model=self._value_or_none(self.vertex_model_var.get()),
            vertex_dim=self._value_or_none(self.vertex_dim_var.get()),
            gemini_model=self._value_or_none(self.gemini_model_var.get()),
            gemini_task_type=self._value_or_none(self.gemini_task_type_var.get()),
            gemini_batch_size=self._value_or_none(self.gemini_batch_size_var.get()),
            embedding_normalize=self.embedding_normalize_var.get(),
            embedding_sections=[name for name, var in self.embedding_sections_vars.items() if var.get()],
            embedding_version=self._value_or_none(self.embedding_version_var.get()),
            classify_ccs=self.classify_ccs_var.get(),
            ccs_model=self._value_or_none(self.ccs_model_var.get()),
            ccs_taxonomy=self._value_or_none(self.ccs_taxonomy_var.get()),
            ccs_max_concepts=self._value_or_none(self.ccs_max_concepts_var.get()),
            ccs_top_candidates=self._value_or_none(self.ccs_top_candidates_var.get()),
            ccs_fallback_candidates=self._value_or_none(self.ccs_fallback_candidates_var.get()),
            ccs_temperature=self._value_or_none(self.ccs_temperature_var.get()),
            ccs_max_output_tokens=self._value_or_none(self.ccs_max_output_tokens_var.get()),
            ccs_embedding_model=self._value_or_none(self.ccs_embedding_model_var.get()),
            openai_key=self._value_or_none(self.openai_key_ui_var.get()),
            gemini_api_key=self._value_or_none(self.gemini_api_key_var.get()),
        )

        try:
            request, config_path = prepare_pipeline_command(form, self.python_exec, REPO_ROOT)
        except ValueError as exc:
            messagebox.showerror("入力エラー", str(exc))
            return

        if config_path:
            self._append_log(f"[INFO] Config written: {config_path}")

        self._execute_async(request)

    def _execute_async(self, request: CommandRequest) -> None:
        def runner() -> None:
            self._is_running = True
            self._append_log(f"[RUN] {' '.join(request.command)}")
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
                for line in process.stdout:
                    self._append_log(line.rstrip())
                return_code = process.wait()
                self._append_log(f"[DONE] return code={return_code}")
            except Exception as exc:  # pragma: no cover - UI surface
                self._append_log(f"[ERROR] {exc}")
                messagebox.showerror("実行エラー", str(exc))
            finally:
                if request.cleanup_paths:
                    for path in request.cleanup_paths:
                        if path and path.exists():
                            try:
                                path.unlink()
                                self._append_log(f"[INFO] Removed temp file: {path}")
                            except OSError:
                                pass
                self._is_running = False

        self._active_thread = threading.Thread(target=runner, daemon=True)
        self._active_thread.start()

    # ------------------------------------------------------------------ embedding only
    def _run_embeddings_only(self) -> None:
        if self._is_running:
            messagebox.showwarning("実行中", "他の処理が実行中です。完了をお待ちください。")
            return
        summary_dir = self.emb_summary_dir_var.get().strip()
        if not summary_dir:
            messagebox.showerror("入力エラー", "サマリー JSON のディレクトリを指定してください。")
            return
        options = EmbeddingToolOptions(
            summary_dir=summary_dir,
            provider=self._value_or_none(self.embedding_provider_var.get()),
            normalize=self.embedding_normalize_var.get(),
            sections=[name for name, var in self.embedding_sections_vars.items() if var.get()],
            embedding_version=self._value_or_none(self.embedding_version_var.get()),
            force=self.embedding_force_var.get(),
            dry_run=self.embedding_dry_run_var.get(),
            output_dir=self._value_or_none(self.emb_output_dir_var.get()),
            env_file=self._value_or_none(self.env_path_var.get()),
            local_model=self._value_or_none(self.local_model_var.get()),
            vertex_project=self._value_or_none(self.vertex_project_var.get()),
            vertex_location=self._value_or_none(self.vertex_location_var.get()),
            vertex_model=self._value_or_none(self.vertex_model_var.get()),
            vertex_dim=self._value_or_none(self.vertex_dim_var.get()),
            gemini_model=self._value_or_none(self.gemini_model_var.get()),
            gemini_api_key=self._value_or_none(self.gemini_api_key_var.get()),
            gemini_task_type=self._value_or_none(self.gemini_task_type_var.get()),
            gemini_batch_size=self._value_or_none(self.gemini_batch_size_var.get()),
            openai_key=self._value_or_none(self.openai_key_ui_var.get()),
        )
        request = prepare_embeddings_command(options, self.python_exec, REPO_ROOT)
        self._execute_async(request)

    # ------------------------------------------------------------------ network helpers
    def _check_grobid_status(self) -> None:
        if self.extractor_var.get() != "grobid":
            messagebox.showinfo("GROBID", "まず GROBID を利用する設定に切り替えてください。")
            return

        url = self.grobid_url_var.get().strip().rstrip("/")
        timeout = self._coerce_float(self.grobid_timeout_var.get()) or 60.0
        endpoint = f"{url}/api/isalive"

        def worker() -> None:
            try:
                request = urllib.request.Request(endpoint, method="GET")
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    code = getattr(response, "status", 200)
                    self.grobid_status_var.set(f"OK (HTTP {code})")
                    self._append_log(f"[INFO] GROBID OK: {endpoint} -> {code}")
            except urllib.error.URLError as exc:
                self.grobid_status_var.set("失敗")
                self._append_log(f"[ERROR] GROBID not reachable: {exc}")
                messagebox.showerror("GROBID", f"ヘルスチェック失敗: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _check_docker_for_grobid(self) -> None:
        def worker() -> None:
            try:
                result = subprocess.run(
                    ["docker", "ps", "--format", "{{.Names}}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
            except FileNotFoundError:
                messagebox.showwarning("Docker", "`docker` コマンドが見つかりません。Docker Desktop などのセットアップを確認してください。")
                self._append_log("[WARN] docker command not found")
                return

            if result.returncode != 0:
                self._append_log(f"[WARN] docker ps failed: {result.stderr.strip()}")
                messagebox.showwarning("Docker", f"`docker ps` の実行に失敗しました: {result.stderr.strip()}")
                return

            containers = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            matching = [name for name in containers if "grobid" in name.lower()]
            if matching:
                messagebox.showinfo("Docker", f"GROBID 関連コンテナが稼働中です: {', '.join(matching)}")
                self._append_log(f"[INFO] GROBID containers detected: {', '.join(matching)}")
            else:
                messagebox.showinfo("Docker", "GROBID 名前を含むコンテナは見つかりませんでした。`docker compose up grobid` を実行してください。")
                self._append_log("[INFO] No running GROBID containers detected.")

        threading.Thread(target=worker, daemon=True).start()

    def _fetch_models(self) -> None:
        token = self.openai_key_ui_var.get().strip() or os.getenv("OPENAI_API_KEY")
        if not token:
            messagebox.showerror("OpenAI Models", "OpenAI API Key を入力するか環境変数に設定してください。")
            return

        search_term = self.model_search_var.get().strip().lower()

        def worker() -> None:
            try:
                request = urllib.request.Request(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {token}"},
                )
                with urllib.request.urlopen(request, timeout=15) as response:
                    body = response.read().decode("utf-8")
            except urllib.error.URLError as exc:
                messagebox.showerror("OpenAI Models", f"モデル取得に失敗しました: {exc}")
                self._append_log(f"[ERROR] Failed to fetch models: {exc}")
                return

            try:
                payload = json.loads(body)
                data = payload.get("data", [])
            except json.JSONDecodeError as exc:
                messagebox.showerror("OpenAI Models", f"API からの応答の解析に失敗しました: {exc}")
                self._append_log(f"[ERROR] Failed to decode models response: {exc}")
                return

            models = sorted(item.get("id", "") for item in data if isinstance(item, dict))
            if search_term:
                models_filtered = [m for m in models if search_term in m.lower()]
            else:
                models_filtered = models

            self.models_listbox.delete(0, tk.END)
            for model_id in models_filtered:
                self.models_listbox.insert(tk.END, model_id)
            self._append_log(f"[INFO] {len(models_filtered)} models listed.")

        threading.Thread(target=worker, daemon=True).start()

    def _apply_selected_model(self) -> None:
        selection = self.models_listbox.curselection()
        if not selection:
            messagebox.showwarning("モデル選択", "一覧からモデルを選択してください。")
            return
        model_id = self.models_listbox.get(selection[0])
        self.summary_model_var.set(model_id)
        messagebox.showinfo("モデル選択", f"サマリーモデルを {model_id} に設定しました。")

    # ------------------------------------------------------------------ utilities
    @staticmethod
    def _coerce_int(value: str) -> Optional[int]:
        value = value.strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    def _coerce_float(value: str) -> Optional[float]:
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _value_or_none(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


def main() -> None:
    root = tk.Tk()
    app = PreprocessingUI(root)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()
