# 前処理パイプラインガイド / Pre-Processing Pipeline Guide

## 概要 / Overview
- このドキュメントは `Pre-Processing` 配下のスクリプト群が提供する PDF 取得・要約・CCS 分類・埋め込み計算の流れをまとめたものです。
- 主要 CLI の役割、依存関係、典型的なコマンド例、設定ポイントを日英併記で整理しています。
- 既存コードの詳細は各モジュール内コメントを参照してください。ここではオペレーション手順と全体像を重視します。

## データフローの全体像 / End-to-End Flow
1. **資料取得 / Fetch PDFs**  
   `download_wiss_pdfs.py` が WISS CSV から PDF を年別に取得します。既存ファイルはスキップまたは `--overwrite` 指定で再取得できます。
2. **ジョブ登録 / Queue Jobs**  
   `orchestrator.py run` が指定した PDF 群を SQLite ベースのジョブストア (`utils/state.py`) に登録します。`--force` で済ジョブを再投入できます。
3. **要約生成 / Summarise**  
   `summary/summarize_pdf.py` が OpenAI Responses API を用いて日本語メイン＋英語翻訳を含む JSON 要約を生成します。必要に応じて GROBID 抽出へフォールバックします。
4. **埋め込み計算 / Embeddings**  
   セクション別ベクトルを `embeddings.py` 経由で Gemini / Vertex AI / Sentence Transformer から取得し、要約 JSON に書き込みます。
5. **ACM CCS 分類 / CCS Classification**  
   `ccs/classifier.py` が要約を英語テキストに整形し、埋め込みベースの候補抽出＋LLM による最終選定を行います。
6. **成果物保存 / Persist Outputs**  
   出力 JSON は `output/summaries/` に保存、ジョブ履歴とログは `data/` と `.cache/` を中心に管理されます。

## 環境準備 / Environment Setup
- Python 3.10 以上を推奨します。仮想環境を作成し依存をインストールしてください。
  ```bash
  python -m venv .venv
  .\.venv\Scripts\activate  # Windows
  pip install -r requirements.txt
  ```
- `.env` には最低限 `OPENAI_API_KEY`, `GEMINI_API_KEY`, `WISS_DATA_ROOT` を記載します。Vertex AI を使う場合は `VERTEX_AI_PROJECT`, `VERTEX_AI_LOCATION`, `VERTEX_AI_EMBEDDING_MODEL` などを追加してください。
- すべての CLI は `utils.env.load_env()` を通して `.env` を自動読込し、`utils.paths.ensure_preprocess_path()` でモジュール解決を行います。
- 文字起こしが難しい PDF には `pypdf`＋`pdfminer.six`、または `--extractor grobid` と外部 GROBID サービスを利用します。

## ディレクトリと主なスクリプト / Key Modules
| パス / Path | 役割 / Role |
| --- | --- |
| `download_wiss_pdfs.py` | WISS CSV から PDF をダウンロードし、年別に整理します。 |
| `orchestrator.py` | PDF 探索→ジョブ登録→要約生成→埋め込み→CCS 分類をワークキューで実行します。並列実行 (`--workers`) と再実行制御を提供します。 |
| `summary/summarize_pdf.py` | 単一 PDF の要約 CLI。チャンク分割・メタデータ推定・日英サマリー生成・翻訳整形を実装しています。 |
| `summary/batch_summarize.py` | 複数 PDF を一括処理する軽量バッチ用ラッパーです。 |
| `pdf_extractors.py` / `summary/pdf_extractors.py` | PyPDF・pdfminer・GROBID を切り替える抽出層を定義します。 |
| `embedding/compute_embeddings.py` | 既存サマリー JSON を読み込み、指定プロバイダで埋め込みを再計算します。Gemini バッチ API や Vertex AI に対応しています。 |
| `embeddings.py` | 各プロバイダ共通のエラー処理、正規化、キャッシュをまとめています。 |
| `ccs/classify_ccs.py` | 要約済み JSON へ ACM CCS ラベルを付与し、必要に応じてファイルを上書きします。 |
| `ccs/classifier.py` | 埋め込みによる候補抽出と LLM プロンプト生成、信頼度計算を行う再利用可能コンポーネントです。 |
| `utils/state.py` | SQLite バックエンドのジョブキューを定義し、`STATUS_*` 遷移や再実行制御 (`mark_running`, `mark_failed` など) を提供します。 |
| `utils/env.py` / `utils/paths.py` | `.env` 探索とパス解決、`WISS_DATA_ROOT` や `WISS_PIPELINE_DB` の既定値管理を行います。 |
| `ui/app.py` | Tkinter の簡易 GUI で GROBID ヘルスチェックや CLI 呼び出しを一括操作できます。 |
| `output/summaries/` | 要約および埋め込み JSON のデフォルト出力先です (存在しない場合は自動生成)。 |
| `data/` | 旧デフォルトの SQLite (`pipeline_state.db`) や補助データを保持します。 |

## 典型的な運用手順 / Typical Workflow

### 1. PDF 取得 / Fetch PDFs
```bash
python download_wiss_pdfs.py --csv-dir WISSProceedings --output-dir data/raw/WISS --years 2023 2024
```
- `--dry-run` でダウンロード予定のみ確認できます。
- `missing_pdfs.csv` に失敗ログが蓄積されます。

### 2. パイプライン実行 / Run Full Pipeline
```bash
python orchestrator.py run ^
  --pdf-dir data/raw/WISS/2024 ^
  --workers 2 ^
  --output-dir output/summaries ^
  --pretty
```
- `--env-file` で特定の `.env` を指定できます。
- `--limit` で処理件数を制限、`--force` で完了済みジョブを再投入します。
- 実行中は標準出力に進捗が表示され、完了後に成功・失敗件数が集計されます。
- ジョブ状態は `python orchestrator.py status` で確認できます。

### 3. 埋め込みの更新 / Refresh Embeddings
```bash
python embedding/compute_embeddings.py output/summaries --provider gemini --model gemini-embedding-001 --force
```
- `--sections positioning purpose` のように対象セクションを限定可能です。
- `--output-dir` を指定すると既存ファイルを上書きせずに別ディレクトリへ保存します。
- Gemini のレート制限に達すると `EmbeddingQuotaExceeded` が発生し、残ジョブはスキップされます。

### 4. CCS 再分類のみ / CCS-only Pass
```bash
python ccs/classify_ccs.py output/summaries/*.json --update --model gpt-5-mini
```
- 既存 JSON に `ccs.predictions` ブロックを追記し、`--output` で JSONL レポートとして別出力も可能です。
- `--embedding-model none` を指定すると埋め込み候補抽出をスキップします (速度重視・精度低下に注意)。

### 5. 個別要約 / Single PDF Quick Run
```bash
python summary/summarize_pdf.py paper.pdf --output output/summaries/paper.json --pretty --embeddings
```
- `--extractor grobid --grobid-url http://localhost:8070` で GROBID を利用できます。
- `--language English --no-dual-language` のように言語や翻訳設定を調整できます。

### 6. GUI からの操作 / Operate via GUI
```bash
python ui/app.py
```
- 仮想環境を検知すると GUI から起動した CLI も同一環境を使用します。
- GROBID のヘルスチェックや `.env` 設定ビューが含まれます。

## 設定とチューニング / Configuration Tips
- `orchestrator.py` は JSON 設定ファイルを `--config` で読み込み、`SummariserConfig` の任意フィールドを上書きできます。例: チャンク長 (`chunk_size`), モデル (`model`), 埋め込みプロバイダ (`embedding_provider`), CCS モデル (`ccs_model`)。
- `output_dir` を設定ファイルに含めると、CLI で `--output-dir` を明示しなくても要約 JSON の出力先を切り替えられます（CLI 引数が優先されます）。/ Including `output_dir` in the config JSON lets you change the summary destination without passing `--output-dir` (explicit CLI flags still win).
- 主要環境変数:
  - `WISS_DATA_ROOT`: PDF を探索するディレクトリ (カンマ区切りで複数指定可)。
  - `WISS_PIPELINE_DB`: SQLite ジョブ DB の保存場所。未指定時はユーザごとのアプリデータ配下に作成されます。
  - `GEMINI_API_KEY` / `OPENAI_API_KEY` / `VERTEX_AI_*`: 各 API キー・プロジェクト設定。
- GROBID 利用時は Docker などでサービスを起動し、`--grobid-timeout` でタイムアウトを調整してください。
- `utils/state.PipelineState.purge_missing_jobs()` を呼び出すと、削除済み PDF に対応するジョブを掃除できます。

## トラブルシューティング / Troubleshooting
- **API キー不足**: `OPENAI_API_KEY` や `GEMINI_API_KEY` が未設定だと即時終了します。`.env` の位置を確認し `python - <<<'import utils.env as e; print(e.locate_env_file())'` などで検出を確認してください。
- **抽出失敗**: PyPDF が空文字を返す場合は pdfminer をインストールするか GROBID に切り替えてください。
- **Gemini クォータ**: エラー発生時は `embedding/compute_embeddings.py` が `quota` ステータスで停止します。時間を置いて `--force` を付けて再実行します。
- **ジョブ詰まり**: 例外で停止したジョブは `STATUS_FAILED` になります。再実行は `--force` もしくは `PipelineState.enqueue(..., force=True)` を利用します。

## 付録 / Appendix
- テスト: `python -m compileall Pre-Processing` で構文チェック、`pytest tests/preprocessing` (存在する場合) でユニットテストを実行できます。
- データ整合性: 要約 JSON の `metadata_meta` フィールドには DOI/年の信頼度と出典が記録されています。下流処理での品質判定に活用できます。
- ローカル LLM モード: `Log/local_llm_mode_direction.md` でローカル推論時の調整メモを管理しています。環境切り替え時は併せて確認してください。

---

運用フローの更新や追加スクリプトが生じた場合は、このファイルを最新版に保ち、`orchestrator.py` や `README.md` からリンクしてください。
