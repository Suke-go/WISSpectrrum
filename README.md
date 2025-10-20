# WISSpectrrum

WISSpectrrum は WISS (Workshop on Interactive Systems and Software) の論文群を対象に、AI 要約と ACM CCS 分類を組み合わせた探索体験を構築するプロジェクトです。PDF の収集から要約・埋め込み生成、概念ツリー化、そしてシェーダ駆動の 3D ビジュアライザーまでを一貫して扱います。

WISSpectrrum combines ingestion, summarisation, embedding, and a shader-driven 3D explorer so that WISS papers can be surveyed, compared, and rediscovered from both semantic distances and ACM CCS concepts.

## 主な機能 / Key Capabilities
- インタラクティブな Three.js バブルシーンで ACM CCS ツリーと論文クラスタを探索 (Interactive bubble scene built with Three.js for navigating the ACM CCS hierarchy and paper clusters)
- PDF → 構造化テキスト → セクション別要約 → Embedding → ACM CCS 分類までを自動化 (End-to-end automation from PDF extraction to section summaries, embeddings, and ACM CCS assignment)
- OpenAI GPT-5 mini 系モデルと Gemini Embedding / Vertex AI / ローカル埋め込みの切り替えに対応 (Supports OpenAI GPT-5 mini for summaries with Gemini, Vertex AI, or local Sentence Transformer embeddings)
- `build_index.py` / `build_enhanced_index.py` による可視化用インデックス生成 (Generates viewer-ready `index.json` and enhanced coordinates for the UI)
- Tkinter / Streamlit コントロールパネルによるパイプライン操作と進捗監視 (Tkinter and Streamlit front-ends make it easy to operate and monitor the pipeline)

## ディレクトリ構成 / Repository Layout
- `Pre-Processing/` – パイプライン本体 (summaries, embeddings, ACM CCS, UI ツール) / Python pipeline and operator tooling
- `frontend/` – Vite + React + Three.js による最終ビジュアライザー / production UI
- `viz/` – 旧プロトタイプとデバッグ用 HTML/JS / legacy prototypes kept for reference
- `thesis/` – 元 PDF を配置する既定パス (リポジトリには含まれません) / expected raw PDF root
- `Pre-Processing/output/summaries/` – 要約 JSON・インデックス・埋め込みの出力先 / generated artefacts
- `MASTER_DOCUMENT.md` – 体験設計・研究背景の詳細メモ / in-depth background document

## 研究文脈 / Research Context
- WISS 論文の探索・比較・再発見を支援する研究基盤として設計し、インタラクション研究における知識整理や調査報告の再現性を高めます。  
  Designed as an academic infrastructure for HCI and interactive systems research, enabling reproducible literature surveys and comparative studies over the WISS proceedings corpus.
- セクション別要約と ACM CCS ラベルを併用することで、問題意識・目標・手法・評価の差異を定性的・定量的に分析可能です。  
  Section-aware summaries combined with ACM CCS tags allow both qualitative and quantitative analyses across problem framing, objectives, methods, and evaluation strategies.
- 生成される JSON / index 群は引用可能なメタデータ (作成日時、処理設定、モデル名) を保持し、研究論文や発表での再利用と再計算を支援します。  
  Generated JSON and indices carry citeable metadata (timestamps, configuration, model versions), supporting reuse and recomputation in academic papers and presentations.
- `thesis/<venue>/<year>/` の入出力構造は複数年度・別領域への横展開を意図しており、追試や拡張研究に必要なファイル体系を明示しています。  
  The prescribed `thesis/<venue>/<year>/` layout generalises to other venues and years, making replication studies and cross-domain extensions straightforward.

## セットアップ手順 / Setup

### 1. Python パイプライン環境
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.env.example` を参考にルート直下に `.env` を作成し、最低限以下を設定します (全 CLI が起動時に自動読み込みします)。

```
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
WISS_DATA_ROOT=thesis
# オプション
# WISS_PIPELINE_DB=%LOCALAPPDATA%/WISSpectrrum/pipeline_state.db
# VERTEX_AI_PROJECT=your-project
# VERTEX_AI_LOCATION=us-central1
# VERTEX_AI_EMBEDDING_MODEL=text-embedding-004
```

GROBID を使う場合は別途 Docker などで起動し、サマライザに `--extractor grobid --grobid-url http://localhost:8070` を指定します。

### 2. 論文 PDF とメタデータの収集
1. 必要なら `Pre-Processing/scraipingpdf/` 配下のスクレイパで WISS Proceedings から CSV を作成します。
2. `Pre-Processing/pdfdownloader/download_wiss_pdfs.py` で CSV から PDF をダウンロードし、`thesis/<venue>/<year>/<paper>.pdf` 形式で保存します。

```bash
python Pre-Processing/pdfdownloader/download_wiss_pdfs.py ^
  --input WISSProceedings/wiss2024.csv ^
  --output-root thesis/WISS/2024 ^
  --delay 1.0
```

### 3. 要約・埋め込み・CCS の一括実行
`orchestrator.py` は PDF を探索してジョブキューに登録し、OpenAI/Gemini などを呼び出して JSON 出力を生成します。

```bash
python Pre-Processing/orchestrator.py run \
  --pdf-dir thesis/WISS/2024 \
  --output-dir Pre-Processing/output/summaries \
  --workers 2 \
  --pretty
```

- `--pdf` / `--pdf-dir` を複数回指定すると対象を絞れます。
- `--config path/to/config.json` で `SummariserConfig` の任意項目を上書きできます。
- ジョブ状態は `python Pre-Processing/orchestrator.py status` で確認できます。
- 処理状況は `Pre-Processing/data/` または `WISS_PIPELINE_DB` で指定した SQLite に永続化されます。

個別 CLI を直接使う場合は `Pre-Processing/summary/summarize_pdf.py` や `Pre-Processing/embedding/compute_embeddings.py` を呼び出してください。

### 4. ビジュアライザー向けインデックス生成
要約が揃ったら以下を実行し、`index.json` と 2D 座標入りの `index_enhanced.json` を生成します。

```bash
python Pre-Processing/output/build_index.py
python Pre-Processing/output/build_enhanced_index.py
```

これらのファイルはフロントエンドが `/Pre-Processing/output/summaries/` (開発時) または `/data/summaries/` (本番環境) から取得します。

### 5. フロントエンドのセットアップ
```bash
cd frontend
npm install
npm run dev
```

開発サーバは `http://localhost:3000` で起動し、上位ディレクトリの `Pre-Processing/output/summaries/` を自動的にプロキシします。データを更新した場合はブラウザをリロードするだけで反映されます。

本番ビルドは `npm run build`。生成される `frontend/dist/` を配信し、同じホスト上の `/data/summaries/` に `index.json`, `index_enhanced.json`, 各年の JSON を配置してください (構成例: `dist/data/summaries/...` を静的ホスティングルートにコピー)。

## データ成果物 / Generated Artefacts

### 個別要約 JSON (`Pre-Processing/output/summaries/<year>/<slug>.json`)
- `title`, `authors` と `*_en` フィールド: 日英のタイトル・著者名・アブストラクト
- セクション要約: `positioning_summary`, `purpose_summary`, `method_summary`, `evaluation_summary` (+ `_en`)
- `embeddings`: モデル名・バージョン・ベクトル (セクション別にも対応)
- `ccs`: `paths`, `ids`, `llm_explanations` など ACM CCS 関連情報
- `links`: PDF / 動画 / 補足資料など既知の URL
- `metadata_meta`, `processing_meta`: 抽出器やリトライ回数などの処理ログ
- `translations`: `_en` を平坦化する前の原データ (オプション)

### `index.json`
- `generated_at`: インデックス生成時刻
- `metadata`: `total_papers`, `total_concepts`, `embedding_version`, `embedding_dimensions` など
- `concept_tree`: ACM CCS をベースにした概念ツリー。各ノードに `count`, `children`, `papers` がぶら下がります
- `years`: 年ごとの論文リスト (`papers[].path` で個別 JSON へリンク)

### `index_enhanced.json`
`index.json` に加え、`papers[].embedding_2d[section].tsne` / `pca` に 2D 座標を格納します。フロントエンドはこれをもとに距離レンズやアニメーションを生成します。

### `pipeline_state.db`
`orchestrator.py` が参照する SQLite。ジョブキュー、実行履歴、リトライ情報を保持します。`WISS_PIPELINE_DB` を設定して別ディスクに移すこともできます。

## 補足ツール / Additional Tooling
- `Pre-Processing/ui/app.py`: Tkinter ベースのコックピット。GROBID のヘルスチェックや要約ジョブ投入を GUI で行えます。
- `Pre-Processing/ui/app_streamlit.py`: Streamlit 版 UI。`streamlit run Pre-Processing/ui/app_streamlit.py` で起動。
- `tests/test_ccs_classifier.py`: ACM CCS 分類器の単体テスト。`pytest` を導入した上で `pytest tests/test_ccs_classifier.py` を実行してください。
- `Log/`, `Presentation/`, `WISSChallengeSubmission/`: プロジェクト記録・資料類 (コード以外)。

## よく使うコマンド / Handy Commands
- `python Pre-Processing/orchestrator.py run --pdf-dir thesis/WISS/2024 --workers 4 --force`
- `python Pre-Processing/orchestrator.py status`
- `python Pre-Processing/embedding/compute_embeddings.py Pre-Processing/output/summaries/2024/*.json --provider gemini --normalize`
- `python Pre-Processing/ccs/classify_ccs.py Pre-Processing/output/summaries/2024/*.json --max-concepts 3`
- `npm run lint`, `npm run test`, `npm run preview` (フロントエンド)

## トラブルシューティング / Troubleshooting
- **GROBID が応答しない**: コンテナが起動しているか、`--grobid-url` が正しいか、`--grobid-timeout` を十分に長く取っているか確認してください。
- **Gemini のクォータ超過**: `EmbeddingQuotaExceeded` が発生した場合は `--embedding-provider local` や Vertex AI を利用するか、API キーのクォータを拡張します。
- **データが UI に表示されない**: `index.json` が最新か、ホスティング環境で `/data/summaries/` が公開されているか、ブラウザのキャッシュをクリアしてください。

さらなる詳細や設計の背景は `Pre-Processing/README.md` と `MASTER_DOCUMENT.md` を参照してください。質問や改善提案があれば Issue や Pull Request でお知らせください。
