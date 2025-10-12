# PDF Processing & Summarisation Workflow

## English

### 0. Prerequisites
- Python 3.10+ is recommended. A ready-made virtual environment lives at `.venv/`; create one with `python3 -m venv .venv` (macOS/Linux) or `py -m venv .venv` (Windows) if it ever goes missing.
- Activate the environment and install dependencies:
  ```bash
  . .venv/bin/activate
  pip install -r requirements.txt  # or install `openai`, `requests`, `pypdf`, `sentence-transformers` as needed
  ```
- Platform notes:
  - **macOS / Linux:** use the `python3` interpreter that matches the virtualenv. If the activate script complains about execution rights, run `chmod +x .venv/bin/activate`.
  - **Windows (PowerShell):**
    ```powershell
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```
    Run scripts with `.\.venv\Scripts\python.exe` instead of `.venv/bin/python`.
- Add your OpenAI (and optional Vertex AI) secrets to `.env` or the shell before invoking the summariser:
  ```bash
  export OPENAI_API_KEY="..."
  ```
- Optional: Docker with enough RAM (≥4 GB) to host GROBID.

### 1. Collect WISS PDFs
- Use the helper script to resolve CSV rows to files:
  ```bash
  .venv/bin/python Pre-Processing/pdfdownloader/download_wiss_pdfs.py \
      --input WISSProceedings/wiss2024.csv \
      --output-root thesis/WISS/2024
  ```
- The summariser expects PDFs at `thesis/<venue>/<year>/<paper>.pdf`. Keep that structure so metadata merges remain deterministic.
- Tweak options if needed:
  - `--dry-run` prints planned downloads without touching disk.
  - `--delay` and `--timeout` control pacing; increase the delay when the host rate-limits you.
  - `--overwrite` lets you refresh previously downloaded files.

### 1.5 Responsible downloading
- Respect the WISS proceedings terms of use; download only what you are allowed to process.
- Keep the default one-second delay (or make it longer) to avoid hammering the server, and prefer `--years` to limit requests.
- Cache the PDFs you fetch—rerun the summariser against local files instead of redownloading.
- Monitor the logs for HTTP errors; repeated 4xx/5xx responses are a signal to pause and investigate.

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
- Python 3.10 以上を推奨します。仮想環境 `.venv/` が無い場合は `python3 -m venv .venv`（macOS/Linux）または `py -m venv .venv`（Windows）で作成してください。
- 仮想環境を有効化し、依存ライブラリをインストール:
  ```bash
  . .venv/bin/activate
  pip install -r requirements.txt  # ない場合は openai, requests, pypdf などを個別に導入
  ```
- 環境ごとのメモ:
  - **macOS / Linux:** `python3` コマンドが仮想環境のバージョンと一致しているか確認してください。`activate` に実行権限が無い場合は `chmod +x .venv/bin/activate` を実行します。
  - **Windows (PowerShell):**
    ```powershell
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```
    スクリプト実行時は `.\.venv\Scripts\python.exe` を使用してください。
- 要約スクリプトを動かす前に `.env` またはシェルで `OPENAI_API_KEY`（必要に応じて Vertex AI の設定）を指定します:
  ```bash
  export OPENAI_API_KEY="..."
  ```
- 任意: GROBID を動かす Docker (メモリ 4GB 以上を推奨)。

### 1. WISS PDF の取得
- CSV に基づいて論文 PDF を保存する場合の例:
  ```bash
  .venv/bin/python Pre-Processing/pdfdownloader/download_wiss_pdfs.py \
      --input WISSProceedings/wiss2024.csv \
      --output-root thesis/WISS/2024
  ```
- `thesis/<学会>/<年度>/<ファイル名>.pdf` という配置に揃えると、要約スクリプト側でメタデータを統合しやすくなります。
- 主なオプション:
  - `--dry-run`: 実際に保存せず予定リストのみ表示。
  - `--delay` / `--timeout`: リクエスト間隔やタイムアウトの調整（混雑時は `--delay` を長めに）。
  - `--overwrite`: 既存ファイルを上書き。

### 1.5 スクレイピング時の注意
- WISS の利用規約を遵守し、権限のある資料のみ取得してください。
- デフォルトの 1 秒ディレイ（またはそれ以上）を守り、`--years` で対象を絞り込むと負荷を抑えられます。
- 取得した PDF はローカルに保存し、再要約時は再ダウンロードせずキャッシュを活用してください。
- HTTP エラー（4xx/5xx）が続く場合は処理を中断し、原因を確認しましょう。

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
