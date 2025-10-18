# Pre-Processing Utilities

このディレクトリには、論文 PDF からの要約生成、ACM CCS 分類、埋め込み計算を行うための CLI がまとまっています。以下では代表的なスクリプトと使い方を紹介します。  
This directory collects CLI tools that summarise research PDFs, classify ACM CCS concepts, and compute embeddings. The sections below walk through the key scripts and usage patterns.

## 更新概要 / Update Summary
- UI: パイプライン操作用の Tkinter フロントエンド `Pre-Processing/ui/app.py` を追加し、GROBID 状態チェックや OpenAI モデル検索を含めた統合操作を提供します。 / Added a Tkinter control panel (`Pre-Processing/ui/app.py`) that unifies pipeline execution, GROBID health checks, and OpenAI model discovery.
- Docs: README と UI 要件メモを日英併記にして、最新の設定項目を把握しやすくしました。 / README and the UI requirement note now include Japanese and English descriptions to highlight the latest configuration options.
- Summary: GPT 要約はトークン不足を自動で検知して再試行し、各セクション 2 文・1 見出しに収まるよう最適化しました。PyPDF で取得できない場合は `pdfminer.six` に自動フォールバックします。 / GPT summarisation now auto-retries when responses truncate and limits section summaries to two sentences per heading, with automatic fallback to `pdfminer.six` when PyPDF cannot extract text.
- Dual-lang: `--dual-language` フラグで英語要約を JSON (`translations.en.*`) に併記し、CCS 分類に活用できます。 / The `--dual-language` flag stores English translations under `translations.en` so CCS classification can rely on English context.

## 0. ディレクトリ構成 / Directory Map
- `summary/`: GPT ベースの要約・CCS 分類ロジック本体。`summary/summarize_pdf.py` が OpenAI Responses API を利用します。 / Core GPT summary + CCS implementation (`summary/summarize_pdf.py`).
- `summarize_pdf.py`: 後方互換 CLI。内部で `summary/summarize_pdf.py` を呼び出します。 / Backwards compatible shim that forwards to `summary/summarize_pdf.py`.
- `embedding/`: 埋め込み計算 CLI の実体。`embedding/compute_embeddings.py` が各プロバイダを束ねます。 / Embedding CLI implementation (`embedding/compute_embeddings.py`).
- `compute_embeddings.py`: 埋め込み CLI のシム。内部で `embedding/compute_embeddings.py` を呼び出します。 / Thin wrapper forwarding to `embedding/compute_embeddings.py`.
- `embeddings.py`: 各種埋め込みバックエンドの共通ヘルパー。 / Shared embedding helpers.
- `utils/`: `.env` ローダーや `paths.py` によるディレクトリ定義、ジョブ状態管理など。 / Utilities such as `.env` loading, path helpers, and job state management.
- `output/summaries/`: 要約・埋め込みの既定保存先（存在しない場合は自動作成されます）。 / Default output directory for summaries & embeddings.
- `data/pipeline_state.db`: バッチ実行時に使う SQLite ジョブキューの既定パス。 / Default SQLite datastore for the orchestrator.
- `ui/`: Tkinter ベースの制御パネル。 / Tkinter control panel for operators.

## 1. 事前準備 / Prerequisites

- Python 3.10+ を想定しています。必要に応じて仮想環境を作成してください。
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- `.env` に API キーを記述すると、各 CLI が起動時に自動で読み込みます。最低限以下を設定してください。
  ```
  OPENAI_API_KEY=xxx
  GEMINI_API_KEY=yyy
  ```
  Vertex AI を使う場合は `VERTEX_AI_PROJECT`, `VERTEX_AI_LOCATION`, `VERTEX_AI_EMBEDDING_MODEL` なども用意します。
- すべての CLI は起動時に `utils.env.load_env()` と `utils.paths.ensure_preprocess_path()` を呼び出し、`.env` の自動探索とモジュール解決を行います。 / Every CLI bootstraps `.env` discovery and sys.path updates via `utils.env` / `utils.paths`.
- `pdfminer.six` をインストールしておくと、PyPDF でテキストが抽出できない PDF に対して自動的にフォールバックします。 / Installing `pdfminer.six` enables automatic fallback extraction when PyPDF cannot read a PDF.
- GROBID を使う場合は Docker などでサービスを起動し、`--extractor grobid --grobid-url http://localhost:8070` を指定してください。 / For GROBID extraction, run the service (e.g. via Docker) and pass `--extractor grobid --grobid-url http://localhost:8070`.

## 2. ワークフロー概要 / Pipeline Overview

1. **PDF 要約 (OpenAI GPT)**: `Pre-Processing/summary/summarize_pdf.py` が PDF からセクション抽出 → チャンク要約 → 最終要約 → ACM CCS 提案までを一括実行します。 / The GPT-based summariser performs extraction → chunk summaries → final synthesis → optional ACM CCS tagging.
2. **埋め込み計算 (Google Gemini)**: `embedding/compute_embeddings.py`（`compute_embeddings.py` シム経由）で JSON 要約の目的・手法等を `gemini-embedding-001` などでベクトル化します。日本語セクションは従来通り保持しつつ、`--dual-language` で英語訳を追加保存できます。 / `embedding/compute_embeddings.py` (reachable via the `compute_embeddings.py` shim) turns summary sections into vectors via Gemini or other providers; `--dual-language` keeps Japanese sections while adding English translations alongside them.
3. **バッチ実行**: `orchestrator.py` や GUI から PDF キューを管理し、要約/埋め込みを `output/summaries/` に蓄積します。 / The orchestrator / GUI manages queues and writes outputs to `output/summaries/`.
4. **後処理オプション**: 既存 JSON に対する CCS 追記や embeddings 再計算を個別 CLI で行えます。 / Follow-up tools enrich existing JSON records.

## 3. PDF 要約 + CCS + 埋め込み

`summarize_pdf.py` は PDF からテキスト抽出 → チャンク要約 → 最終要約 → CCS 分類 → 埋め込み計算までを一括実行します。

```bash
python Pre-Processing/summarize_pdf.py thesis/WISS/2005/fingering_paper.pdf \
  --output summaries/fingering.json \
  --language Japanese \
  --dual-language \
  --classify-ccs \
  --ccs-embedding-model gemini-embedding-001 \
  --embeddings \
  --embedding-provider gemini \
  --embedding-model gemini-embedding-001 \
  --gemini-task-type SEMANTIC_SIMILARITY \
  --section-embeddings
```

- `--classify-ccs` で ACM CCS 分類を有効化します。XML は既定で `Pre-Processing/ACM CCS/acm_ccs2012-1626988337597.xml` を参照します。
- `--ccs-embedding-model` に Gemini もしくは SentenceTransformer の名前を指定します。`gemini-embedding-001` の場合は `.env` の `GEMINI_API_KEY` が利用されます。
- `--embeddings` は要約（purpose/method/evaluation）の埋め込みを、`--section-embeddings` は GROBID などで抽出した各セクションの埋め込みを追加で計算します。Gemini を使う場合は `--embedding-provider gemini --embedding-model gemini-embedding-001` を指定し、必要に応じて `--gemini-task-type` や `--gemini-batch-size` を調整します。
- `.env` と同階層で実行すると自動的にキーが読み込まれます。別ファイルを使う場合は `--env-file` でパスを指定してください。

## 4. 既存要約への ACM CCS 付加

`summaries/*.json` のような既存要約ファイルに後から CCS を付与するには `ccs/classify_ccs.py` を使用します。

```bash
python Pre-Processing/ccs/classify_ccs.py summaries/test_fingering.json \
  --xml Pre-Processing/ACM\ CCS/acm_ccs2012-1626988337597.xml \
  --embedding-model gemini-embedding-001 \
  --update
```

- `--update` を付けると入力ファイルを直接書き換えます。結果のみ確認したい場合はオプションを外してください。
- `.env` を自動で読み込みますが、明示的に指定したい場合は `--env-file path/to/.env` を併用できます。
- Gemini ではなくローカル SentenceTransformer を使う場合は `--embedding-model sentence-transformers/all-MiniLM-L6-v2` のように指定します。

## 5. 埋め込みのみを後付けする

要約が既にあり、埋め込みだけ計算したい場合は `compute_embeddings.py` を利用します。

```bash
python Pre-Processing/compute_embeddings.py summaries/test_fingering.json \
  --sections purpose method evaluation \
  --model gemini-embedding-001 \
  --provider gemini \
  --env-file .env \
  --update
```

- `--provider` で `local`（SentenceTransformer）、`vertex-ai`、`gemini` を選択します。
- `--update` により入力ファイルへ埋め込み情報を追記します。
- `--env-file` を指定すると明示した `.env` を先に読み込みます。指定がない場合は実行ディレクトリから親方向に自動探索します。 / `--env-file` forces a specific environment file; otherwise discovery walks up from the current directory.

## 6. バッチ処理（オプション）

大量の PDF を順番に処理したい場合は `orchestrator.py` を使うと、SQLite を用いたキュー管理と再開が可能です。

```bash
python Pre-Processing/orchestrator.py enqueue data/papers --glob "*.pdf"
python Pre-Processing/orchestrator.py run --classify-ccs --embeddings --section-embeddings
```

キュー操作や各種オプションは `python Pre-Processing/orchestrator.py --help` を参照してください。

## 7. GUI フロントエンド（試験的機能）

Tkinter ベースの簡易 UI `ui/app.py` を用意しています。主要なオプションをまとめて設定し、`orchestrator.py` や `compute_embeddings.py` を直接呼び出します。

```bash
python Pre-Processing/ui/app.py
```

- `.env` の読み込み、GROBID ヘルスチェック、OpenAI モデル一覧取得（`models` API 利用）などを UI 上から実行できます。
- 実行ログは画面下部にストリーム表示され、完了後に結果が確認できます。
- 今後の改善予定: 要約プロンプトのきめ細かな調整や再要約フローの統合など。

## 8. 運用チェックリスト / Operational Checklist
- `.env` を最新の API キーで更新 (`OPENAI_API_KEY`, `GEMINI_API_KEY` 他)。
- `.venv/bin/python` など仮想環境の Python から CLI を実行し、確実に依存を揃えます。
- `pip install -r requirements.txt` で依存を満たし、必要なら `google-generativeai`, `openai`, `pypdf`, `pdfminer.six` などを再確認。
- サマリーツール実行前に `output/summaries/` と `data/` の書き込み権限を確認。無ければ自動作成されます。
- PyPDF で抽出できない場合に備え `pdfminer.six` を導入し、スキャン PDF は OCR（例: `ocrmypdf`）後に投入するか、GROBID ベースの抽出（`--extractor grobid`）に切り替えます。
- GUI (`ui/app.py`) も `.venv/bin/python` が見つかれば自動で採用し、UI からのパイプライン実行・埋め込み計算でも仮想環境依存がそのまま使われます。
- CCS 精度を高めたい場合は `--dual-language` で英語訳を生成し、必要に応じて英語側セクションで埋め込みや分類を行います。
- 単体テスト: `python -m compileall Pre-Processing` でシンタックスチェック、`--dry-run` で書き込み無し実行。
- バッチ処理では `orchestrator.py list` でジョブ状態を確認し、必要に応じて `enqueue --force` や `run --limit` を併用。

---

不明点があれば `README.md`（リポジトリ直下）も併せて確認するか、開発チームまでお問い合わせください。
