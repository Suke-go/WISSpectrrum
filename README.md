# PDF Processing & Summarisation Workflow

## English

### 0. Prerequisites
- Python 3.13 virtual environment lives at `.venv/`. Activate it or call `.venv/bin/python` explicitly.
- Install runtime deps inside the venv:
  ```bash
  . .venv/bin/activate
  pip install -r requirements.txt  # or install `openai`, `requests`, `pypdf`, `sentence-transformers` as needed
  ```
- Optional: Docker with enough RAM (≥4 GB) to host GROBID.

### 1. Collect WISS PDFs
- Use the helper script to resolve CSV rows to files:
  ```bash
  .venv/bin/python Pre-Processing/download_wiss_pdfs.py \
      --input WISSProceedings/wiss2024.csv \
      --output-root thesis/WISS/2024
  ```
- The summariser expects PDFs at `thesis/<venue>/<year>/<paper>.pdf`. Keep that structure so metadata merges remain deterministic.

### 2. Stand up GROBID (recommended extractor)
- Pull and run the service with container JVM tweaks so it boots under cgroup v2:
  ```bash
  docker pull lfoppiano/grobid:0.7.2
  docker run --rm -p 8070:8070 \
      -e JAVA_TOOL_OPTIONS="-XX:-UseContainerSupport" \
      lfoppiano/grobid:0.7.2
  ```
- Health check (from the machine that will run summarisation):
  ```bash
  curl http://<host>:8070/api/isalive   # expect {"grobid":"alive"}
  ```
  Use the host’s LAN IP instead of `localhost` if you run the summariser inside a container/sandbox.

### 3. Generate a summary
- Core command:
  ```bash
  .venv/bin/python Pre-Processing/summarize_pdf.py \
      thesis/WISS/2024/001_parasights-2.pdf \
      --extractor grobid \
      --grobid-url http://<host>:8070 \
      --grobid-timeout 60 \
      --language Japanese \
      --pretty \
      --output summaries/wiss2024_001.json
  ```
- Without GROBID, replace `--extractor grobid` with `--extractor pypdf` (ensure `pypdf` is installed).
- Useful overrides:
  - `--title`, `--author`, `--year`, `--pdf-link`, `--code-link`
  - `--ccs-path`, `--ccs-id` (repeatable)
  - `--chunk-size`, `--overlap` (default 2500/250 chars)
  - `--embeddings --embedding-provider local` (produces vectors for purpose/method/evaluation summaries)

### 4. Output anatomy
- JSON schema (see `synthesize_record`):
  ```jsonc
  {
    "id": "...",                     // DOI or manual id
    "title": "...",
    "authors": ["..."],
    "abstract": "...",
    "positioning_summary": "...",    // problem framing
    "purpose_summary": "...",        // research question or goal
    "method_summary": "...",         // proposed system/method
    "evaluation_summary": "...",     // evaluation procedure + results
    "year": 2024,
    "ccs": {"paths": ["..."], "ids": ["..."]},
    "links": {"pdf": "...", "code": "..."},
    "embeddings": {...}              // optional vectors if requested
  }
  ```
- Summaries are two sentences each (max three), tuned for quick comparison across papers.

### 5. Troubleshooting
- `GROBID service not reachable`: run the container, expose it on a reachable IP, increase `--grobid-timeout`.
- `requests package missing`: install inside the venv (`pip install requests`).
- `pypdf import error`: install `pypdf` or `PyPDF2`.
- Large PDFs causing timeouts: raise `--grobid-timeout` and consider `docker run` with `-m` to allocate more RAM.

---

## 日本語

### 0. 事前準備
- Python仮想環境は `.venv/` にあります。`source .venv/bin/activate` で有効化するか、`.venv/bin/python` を直接使用します。
- 依存ライブラリをインストール:
  ```bash
  . .venv/bin/activate
  pip install -r requirements.txt  # ない場合は openai, requests, pypdf などを個別に導入
  ```
- 任意: GROBID を動かす Docker (メモリ 4GB 以上を推奨)。

### 1. WISS PDF の取得
- CSV に基づいて論文 PDF を保存する場合の例:
  ```bash
  .venv/bin/python Pre-Processing/download_wiss_pdfs.py \
      --input WISSProceedings/wiss2024.csv \
      --output-root thesis/WISS/2024
  ```
- `thesis/<学会>/<年度>/<ファイル名>.pdf` という配置に揃えると、要約スクリプト側でメタデータを統合しやすくなります。

### 2. GROBID の起動（推奨）
- Docker での起動例:
  ```bash
  docker pull lfoppiano/grobid:0.7.2
  docker run --rm -p 8070:8070 \
      -e JAVA_TOOL_OPTIONS="-XX:-UseContainerSupport" \
      lfoppiano/grobid:0.7.2
  ```
- 動作確認:
  ```bash
  curl http://<ホストIP>:8070/api/isalive   # {"grobid":"alive"} が返ればOK
  ```
  サマライザが別環境で動く場合は、`localhost` ではなくホストのIPを指定してください。

### 3. 要約生成
- 代表的なコマンド:
  ```bash
  .venv/bin/python Pre-Processing/summarize_pdf.py \
      thesis/WISS/2024/001_parasights-2.pdf \
      --extractor grobid \
      --grobid-url http://<ホストIP>:8070 \
      --grobid-timeout 60 \
      --language Japanese \
      --pretty \
      --output summaries/wiss2024_001.json
  ```
- GROBID を使わない場合は `--extractor pypdf` を指定します（事前に `pypdf` を導入）。
- 主なオプション:
  - `--title`, `--author`, `--year`, `--pdf-link`, `--code-link`
  - `--ccs-path`, `--ccs-id`（複数指定可）
  - `--chunk-size` / `--overlap`（既定値 2500 / 250 文字）
  - `--embeddings`（目的・手法・評価の埋め込みベクトルを出力）

### 4. 出力 JSON
- 生成される JSON のイメージ:
  ```jsonc
  {
    "id": "WISS2024-001",
    "title": "ParaSights：…",
    "authors": ["..."],
    "abstract": "...",
    "positioning_summary": "...",
    "purpose_summary": "...",
    "method_summary": "...",
    "evaluation_summary": "...",
    "year": 2024,
    "ccs": {"paths": ["..."], "ids": ["..."]},
    "links": {"pdf": "...", "code": "..."}
  }
  ```
- 各 summary フィールドは 2 文（最大 3 文）で、課題・目的・手法・評価＋結果を素早く把握できるようにしています。

### 5. よくあるトラブル
- `GROBID service not reachable`: Docker コンテナが起動しているか、IP/ポートがアクセス可能か、`--grobid-timeout` が短すぎないか確認。
- `requests が見つからない`: 仮想環境内で `pip install requests` を実行。
- `pypdf が見つからない`: `pip install pypdf` もしくは `PyPDF2` を導入。
- 大きな PDF でタイムアウトする: `--grobid-timeout` を増やし、必要なら `docker run` でメモリ上限 (`-m`) を引き上げてください。

エンジニア視点でパイプライン全体を把握できるように設計しているので、構成やフラグを調整しつつ、WISS 論文の比較・整理に活用してください。
