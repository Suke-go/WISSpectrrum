# Pre-Processing Utilities

WISSpectrrum の前処理パイプラインを構成する CLI 群とサポートツールのハブです。PDF を収集・解析し、セクション別サマリー、埋め込みベクトル、ACM CCS 分類、検索インデックス、ビジュアライザー用の `index.json` を生成するまでをこのディレクトリで扱います。

This folder hosts the ingestion pipeline: PDF extraction, section-level summarisation, embeddings, ACM CCS classification, search indices, and UI helpers that feed the WISSpectrrum visualiser.

---

## クイックスタート / Quick Start
1. 仮想環境をアクティベートし依存を導入 (ルートの `README.md` の手順参照)。
2. `.env` に API キー (`OPENAI_API_KEY`, `GEMINI_API_KEY`, …) を設定。
3. 必要な PDF を `thesis/<venue>/<year>/*.pdf` に配置。
4. 一括実行:
   ```bash
   python Pre-Processing/orchestrator.py run \
     --pdf-dir thesis/WISS/2024 \
     --output-dir Pre-Processing/output/summaries \
     --workers 2 \
     --pretty
   python Pre-Processing/output/build_index.py
   python Pre-Processing/output/build_enhanced_index.py
   ```
- build_enhanced_index.py は Annoy ベースの近似最近傍グラフと 2D 座標を前計算し、ビジュアライザーが即時に類似論文を提示できるようにします。実行前に pip install annoy を実行してください。
5. 生成された `Pre-Processing/output/summaries/index.json` と各年の JSON をフロントエンドに渡す。

---

## ディレクトリマップ / Directory Map
- `summary/` – PDF 抽出・要約本体 (`summarize_pdf.py`, `batch_summarize.py`, `pdf_extractors.py`)
- `embedding/` – 埋め込み計算 (`compute_embeddings.py`) とプロバイダ共通ロジック
- `ccs/` – ACM CCS タクソノミーと分類器 (`classifier.py`, `taxonomy.py`)
- `search/` – 埋め込み索引用ユーティリティ (`build_embedding_index.py`, `search_embeddings.py`)
- `output/` – ビジュアライザー向けインデックス生成 (`build_index.py`, `build_enhanced_index.py`)
- `pdfdownloader/` & `scraipingpdf/` – Proceedings からの CSV 生成と PDF ダウンロード
- `ui/` – Tkinter / Streamlit ベースのオペレーター UI
- `utils/` – `.env` ロード、パス解決、ジョブキュー (`utils.state.PipelineState`) 等の共通処理
- `data/`, `output/summaries/` – パイプラインが生成する成果物 (自動作成されます)

---

## 環境と依存関係 / Environment & Dependencies
- Python 3.10+ を推奨。ルート直下で仮想環境を作成し `pip install -r requirements.txt`。
- `.env` で最低限必要なキー:
  - `OPENAI_API_KEY` – GPT-5 mini 系 (要約・ACM CCS) に使用
  - `GEMINI_API_KEY` – 既定の埋め込み (Gemini) で使用
  - `WISS_DATA_ROOT` – PDF 配置ルート (`thesis` が標準値)
  - オプション: `WISS_PIPELINE_DB`, `VERTEX_AI_PROJECT`, `VERTEX_AI_LOCATION`, `VERTEX_AI_EMBEDDING_MODEL`, `GROBID_URL` など
- 追加パッケージ:
  - `pdfminer.six` – PyPDF が失敗した場合のフォールバック抽出
  - `sentence-transformers` / `torch` – ローカル埋め込みプロバイダを利用する場合
  - Annoy – build_enhanced_index.py の近似最近傍計算に使用します (pip install annoy)
  - `google-cloud-aiplatform` – Vertex AI embeddings を使う場合
  - `streamlit` – UI 起動用 (requirements に含まれています)
- 外部サービス:
  - GROBID (optional) – `docker run --rm -p 8070:8070 lfoppiano/grobid:0.7.2`
  - OpenAI Responses API / Gemini Embedding API / Vertex AI

---

## 研究利用の指針 / Academic Usage Notes
- **再現性の確保 (Reproducibility):** 各 CLI は入力パス・出力パス・モデル名・パラメータを JSON / SQLite に記録します。論文執筆時は `processing_meta`, `embedding_meta`, `ccs` セクションを引用し、同じ設定で再計算できるようにしてください。
- **引用可能な成果物 (Citeable Artefacts):** `output/summaries/<year>/<slug>.json` と `index.json` には生成時刻 (`generated_at`) と使用モデル (`model`, `embedding_provider`) が含まれます。研究論文やポスターではこれらを参照情報として明記できます。
- **データ倫理 (Data Ethics):** WISS Proceedings の利用規約を守り、機密資料が含まれる場合はダウンロード対象から除外してください。個人情報が検出された場合はサマリー JSON を再生成し、該当フィールドをマスクしてください。
- **再分析と拡張 (Secondary Analyses):** JSON 形式の成果物は R / Python / Observable などの分析環境に読み込めます。追加の指標や可視化を論文に掲載する際は、変換スクリプトとデータ版数を添付すると査読時の確認が容易になります。

---

## コア CLI / Core Command-Line Interfaces

### `orchestrator.py`
ジョブキューを介して PDF から要約・埋め込み・CCS 分類を一括実行します。

```bash
python Pre-Processing/orchestrator.py run \
  --pdf-dir thesis/WISS/2023 \
  --workers 4 \
  --config configs/summariser.json \
  --force
```

- `--pdf`, `--pdf-dir`, `--pattern`: 入力のフィルタリング
- `--config`: JSON で `SummariserConfig` の初期値を上書き
- `--model`, `--language`, `--embedding-provider`: 一時的なオーバーライド
- `--disable-embeddings`, `--disable-ccs`: 処理ステップをスキップ
- `--workers`: 並列実行数 (API レート制限に注意)
- `status` サブコマンドでジョブ数を集計

### `summary/summarize_pdf.py`
単一 PDF から要約 JSON を生成します (orchestrator から内部的に使用)。

```bash
python Pre-Processing/summary/summarize_pdf.py \
  thesis/WISS/2024/002_hideye-hmd.pdf \
  --extractor grobid \
  --grobid-url http://localhost:8070 \
  --language Japanese \
  --dual-language \
  --output Pre-Processing/output/summaries/2024/002_hideye-hmd.json
```

主なオプション:
- `--chunk-size`, `--overlap`: セクション化の粒度
- `--metadata-chars`: 先頭抽出テキストの長さ (LLM へ渡すメタ情報)
- `--dual-language`, `--flatten-translations`: 英語サマリーの格納方法
- `--classify-ccs`: 生成直後に ACM CCS を付与
- `--compute-embeddings`: 要約本文からセクション別ベクトルを生成

### `embedding/compute_embeddings.py`
既存サマリー JSON に対して埋め込みを再計算します。

```bash
python Pre-Processing/embedding/compute_embeddings.py \
  Pre-Processing/output/summaries/2024/*.json \
  --provider gemini \
  --model gemini-embedding-001 \
  --normalize \
  --force
```

- `--provider`: `gemini` | `vertex-ai` | `local`
- `--sections positioning purpose method evaluation`: セクション限定
- `--output-dir refreshed/` で結果を別フォルダに保存
- `--force` で既存ベクトルを上書き

### `ccs/classify_ccs.py`
サマリー JSON に ACM CCS 概念を後付けします。

```bash
python Pre-Processing/ccs/classify_ccs.py \
  Pre-Processing/output/summaries/2024/*.json \
  --model gpt-5-mini \
  --embedding-model gemini-embedding-001 \
  --top-candidates 20 \
  --max-concepts 3 \
  --output Pre-Processing/output/classifications_2024.jsonl
```

- 既存の `ccs` フィールドがある場合はスキップ。`--force` で上書き
- LLM へのプロンプトは `summary_to_prompt_text` を参照

### `output/build_index.py`
年次 / 概念ツリー / 類似度メタデータを集約して `index.json` を生成します。

```bash
python Pre-Processing/output/build_index.py \
  --summaries-dir Pre-Processing/output/summaries \
  --output Pre-Processing/output/summaries/index.json
```

### `output/build_enhanced_index.py`
index.json を読み込み、セクション毎の埋め込みから PCA + t-SNE で 2D 座標を付与し、Annoy による近似最近傍リストも前計算して index_enhanced.json を出力します。

### `search/build_embedding_index.py` & `search/search_embeddings.py`
- `build_embedding_index.py` – summaries からベクトル検索用 JSONL / Faiss ファイルを生成
- `search_embeddings.py` – 類似論文検索 CLI。`--query-json` で既存サマリーをクエリに再利用可能

### `search/fetch_external_paper_data.py`
外部ソース (ACM DL など) からメタデータを補完するサンプル。`--id`, `--doi`, `--acm` などで取得対象を指定します。

---

## UI ツール / Operator Interfaces
- `ui/app.py` – Tkinter ベース。仮想環境検出、GROBID ヘルスチェック、ジョブ投入、Gemini レート監視などを 1 画面で操作。
- `ui/app_streamlit.py` – Streamlit ダッシュボード。`streamlit run Pre-Processing/ui/app_streamlit.py` でブラウザ UI として利用可能。
- UI からも `.env` が読み込まれ、同じ設定を共有します。

---

## データレイアウト / Data Layout
- 入力 PDF: `WISS_DATA_ROOT` 直下に `<venue>/<year>/<slug>.pdf`
- 出力 JSON: `output/summaries/<year>/<slug>.json`
- インデックス: `output/summaries/index.json`, `index_enhanced.json`
- ジョブログ: `data/pipeline_state.db` (既定)。`.env` の `WISS_PIPELINE_DB` で場所を変更可能
- 追加資料: `output/summaries/*.json` の `links` に任意 URL を格納可能 (UI が参照します)

---

## 運用 Tips / Operational Tips
- 速度調整: `orchestrator.py run --workers N --chunk-size 3000` でバッチサイズを調整。API 制限に合わせてワーカー数を調整してください。
- エラー復旧: `STATUS_FAILED` のジョブは `--force` で再キュー、または `PipelineState.mark_queued` を使った手動復旧が可能。
- ロギング: CLI は標準出力 / エラーで `[INFO]`, `[WARN]`, `[ERROR]` を出力。長期運用では `python -m pip install rich` でカラーログを有効化するオプションもあります (`rich` を import すれば自動で装飾)。
- 大容量 PDF: `--extractor grobid` + `--grobid-timeout 120` で安定化。OCR が必要な場合は `ocrmypdf` などで事前処理。
- Gemini Limit: `EmbeddingQuotaExceeded` が出た場合は `--provider vertex-ai` または `--provider local` に切り替え。
- キャッシュ: サマリー JSON に `processing_meta.embedding_provider` を記録しているので、異なるプロバイダに変えた際は `--force` を忘れずに。

---

## テスト / Testing
- `pytest tests/test_ccs_classifier.py` – ACM CCS 分類の回帰テスト
- `python -m compileall Pre-Processing` – シンタックスチェック
- `pyproject.toml` が無いので `ruff` 等を導入する際は `pip install` 後、個別に実行してください

---

## よくある質問 / FAQ
- **GROBID を使わなくても良いですか?**  
  はい。既定では PyPDF + 簡易ヘッダ推定で JSON を生成します。品質を高めたい場合に GROBID を指定してください。
- **途中で API キーを変更したい**  
  `.env` を更新し、再度コマンドを実行すれば新しいキーが読み込まれます。`orchestrator.py` 実行中に変更した場合は次のジョブから反映されます。
- **LLM 応答が途切れる**  
  `SummariserConfig.chunk_max_tokens` / `final_max_tokens` を増やすか、`--chunk-size` を減らしてチャンク当たりのトークン数を調整してください。

---

必要に応じて `MASTER_DOCUMENT.md` も併読し、体験設計やデータスキーマの背景を確認してください。改善アイデアやバグ報告は Issue / Pull Request で歓迎します。
