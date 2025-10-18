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
- If you still need the proceedings CSV, use the scrapers under `Pre-Processing/scraipingpdf/`. Recent sites work with:
  ```bash
  .venv/bin/python Pre-Processing/scraipingpdf/scraiping2.py \
      https://www.wiss.org/WISS2024Proceedings/ \
      -o WISSProceedings/wiss2024.csv
  ```
  For older layouts (e.g. 2004–2020) you can pass multiple URLs or a text file to `scraiping.py`:
  ```bash
  .venv/bin/python Pre-Processing/scraipingpdf/scraiping.py \
      -f urls.txt \
      -o WISSProceedings/wiss2004-2009.csv
```
  Install scraper dependencies inside the virtualenv if needed (`pip install requests beautifulsoup4 lxml`). The CSV rows include `id`, `title`, `authors`, and absolute links for PDF/video/review assets.
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
  - `--max-workers` enables parallel downloads with a shared rate limiter—raise it only when the source permits concurrent access.
  - `--overwrite` lets you refresh previously downloaded files.

### 1.5 Responsible downloading
- Respect the WISS proceedings terms of use; download only what you are allowed to process.
- Keep the default one-second delay (or make it longer) to avoid hammering the server, and prefer `--years` to limit requests.
- Cache the PDFs you fetch—rerun the summariser against local files instead of redownloading.
- Monitor the logs for HTTP errors; repeated 4xx/5xx responses are a signal to pause and investigate.
- Check the host’s `robots.txt` and published policies before crawling, and stop immediately if access is disallowed.
- Identify yourself with a polite user agent if you fork the downloader (add a `urllib.request.Request` header); the CLI defaults to one worker, so only raise `--max-workers` when the host explicitly allows parallel requests.
- Do not bypass paywalls or authentication flows; obtain consent when in doubt and keep scraped data private if licensing requires it.

### 2. Stand up GROBID (recommended extractor)
- [GROBID](https://grobid.readthedocs.io/) is an open-source service that converts research PDFs into structured TEI XML, providing cleaner text and metadata than lightweight extractors.
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
REA- Without GROBID, replace `--extractor grobid` with `--extractor pypdf` (ensure `pypdf` is installed). The PyPDF path now applies simple heading heuristics so later stages still receive section boundaries; logs will flag any degraded extraction so you can spot PDFs that need manual attention.
- Useful overrides:
  - `--title`, `--author`, `--year`, `--pdf-link`, `--code-link`
  - `--ccs-path`, `--ccs-id` (repeatable)
  - `--chunk-size`, `--overlap` (default 2500/250 chars)
  - `--temperature` to tune sampling (ignored when the selected model is in the GPT-5 family)
  - `--no-dual-language` / `--no-flatten-translations` if you want to opt out of the English fields that are emitted by default (see the JSON schema below).
  - `--embeddings --embedding-provider local` (produces vectors for each summary section — positioning/purpose/method/evaluation and the abstract when available)
  - `--embeddings --embedding-provider vertex-ai --embedding-model text-embedding-004` (uses the same section-wise output backed by Vertex AI embeddings)

### 3.5 Summary style
- The script sends the cleaned body text to the OpenAI Responses API with a constrained prompt that yields four labelled sections: positioning, purpose, method, and evaluation.
- Each section is capped at two concise sentences (three at most) so you can skim dozens of papers quickly while keeping core contributions intact.
- The `--language` flag switches output prose and heading labels between Japanese and English while leaving JSON keys stable for downstream tooling.
- Japanese output swaps the headings for `概要`, `研究の位置づけ／目的`, `ソリューション／エンジニアリング`, and `評価`, matching the `JAPANESE_HEADINGS` map.
- Final synthesis now produces Japanese and English copies of every major field; use `--no-dual-language` to emit Japanese-only JSON or `--no-flatten-translations` to keep the English payload under `translations.en` instead of top-level `*_en` keys.
- When the source mentions metrics, datasets, or participant counts, those details are preserved; if something is missing the field falls back to `"Not specified"`.
- Section detection prefers GROBID-parsed structure; when unavailable it falls back to chunked paragraphs from PyPDF extraction.

### 3.6 Batch orchestrator (new)
- Run the full pipeline—extraction, summarisation, embeddings, and ACM CCS classification—with one command. The orchestrator manages retries via a lightweight SQLite queue:
  ```bash
  .venv/bin/python Pre-Processing/orchestrator.py run \
      --pdf-dir thesis/WISS/2024 \
      --output-dir summaries/wiss2024 \
      --env-file .env \
      --pretty
  ```
- The script enqueues every PDF it discovers (matching `*.pdf` by default) and writes JSON next to the specified output directory, keeping the source subdirectories (e.g. `summaries/wiss2024/001_parasights-2.json`). Re-run with `--force` to regenerate completed jobs.
- Use a JSON config to override summariser settings—store it under version control for reproducibility:
  ```json
  {
    "model": "gpt-5-mini",
    "language": "Japanese",
    "chunk_size": 2800,
    "embedding_provider": "gemini",
    "gemini_embedding_model": "gemini-embedding-001",
    "classify_ccs": true
  }
  ```
  Pass the file with `--config configs/pipeline_wiss2024.json`. CLI flags such as `--model`, `--language`, `--embedding-provider`, `--disable-embeddings`, and `--disable-ccs` override both the defaults and config file.
- Check queue progress or crash counts at any time:
  ```bash
  python Pre-Processing/orchestrator.py status --db data/pipeline_state.db
  ```

### 4. Output anatomy
- JSON schema (see `synthesize_record`):
  ```jsonc
  {
    "id": "...",                          // DOI or manual id
    "title": "...",                       // Japanese
    "title_en": "...",                    // English (auto-generated unless --no-dual-language)
    "authors": ["..."],
    "authors_en": ["..."],
    "abstract": "...",
    "abstract_en": "...",
    "positioning_summary": "...",         // problem framing (JA)
    "positioning_summary_en": "...",      // English copy
    "purpose_summary": "...",             // research question or goal (JA)
    "purpose_summary_en": "...",
    "method_summary": "...",              // proposed system/method (JA)
    "method_summary_en": "...",
    "evaluation_summary": "...",          // evaluation procedure + results (JA)
    "evaluation_summary_en": "...",
    "year": 2024,
    "ccs": {"paths": ["..."], "ids": ["..."]},
    "links": {"pdf": "...", "code": "..."},
    "metadata_meta": {                  // provenance + consistency checks for key fields
      "doi": {
        "status": "consistent|conflict|missing",
        "confidence": "high|medium|low|unknown",
        "preferred_source": "grobid|metadata_llm|synthesized|cli|null",
        "sources": {"grobid": "...", "metadata_llm": "...", "synthesized": "...", "cli": "..."},
        "final": "..."
      },
      "year": { "...": "..." }
    },
    "translations": {                   // present unless --no-dual-language / --no-flatten-translations
      "en": { "title": "...", "abstract": "...", "positioning_summary": "...", "...": "..." }
    },
    "embeddings": {                     // optional; section-wise vectors & metadata when requested
      "provider": "...",
      "model": "...",
      "dim": 768,
      "normed": true,
      "sections": {
        "positioning": [...],
        "purpose": [...],
        "method": [...],
        "evaluation": [...],
        "abstract": [...]
      }
    }
  }
  ```
- Summaries are two sentences each (max three), tuned for quick comparison across papers.

### 4.5 Semantic search (optional)
- Turn a directory of summaries into a per-section embedding index:
  ```bash
  .venv/bin/python Pre-Processing/search/build_embedding_index.py \
      summaries/*.json \
      -o search/embedding_index.jsonl
  ```
  The script keeps only sections with meaningful text (≥40 characters by default) and records their metadata alongside the embedding vector.
- Query the index with a free-form prompt. By default it picks the provider/model stored in the index; override them when multiple combinations exist:
  ```bash
  .venv/bin/python Pre-Processing/search/search_embeddings.py \
      search/embedding_index.jsonl \
      --query "shared AR experiences for elderly care" \
      --top-k 5 \
      --provider vertex-ai \
      --vertex-project YOUR_GCP_PROJECT \
      --vertex-location us-central1
  ```
- If you omit `--vertex-project`/`--vertex-location`, the CLI falls back to `VERTEX_AI_PROJECT`, `GOOGLE_CLOUD_PROJECT`, and `VERTEX_AI_LOCATION` environment variables (default location: `us-central1`).
- When you built embeddings with the local Sentence Transformer backend, drop the Vertex flags—the CLI will reuse the same model automatically.
- To reuse an existing summary as the query, point `--query-json` at a summariser output; each section (positioning/purpose/method/evaluation/abstract) is compared independently and the top matches are displayed per section.
- Pass `--exclude-paper PAPER_ID` to drop specific records (the query JSON's `id` is skipped automatically so you don’t just rediscover the source paper).
- Use the JSONL index as input for downstream dashboards or plug it into vector stores (Pinecone, Weaviate) without re-running the summariser.

### 4.6 ACM CCS classification (LLM-assisted)
- Export (or inspect) the CCM taxonomy from the official SKOS XML. The helper CLI works without network access and provides fuzzy lookup for prompt crafting:
  ```bash
  python Pre-Processing/ccs/export_taxonomy.py --prompt-catalog > ccs_catalog.txt
  python Pre-Processing/ccs/export_taxonomy.py --search "virtual reality haptics" --top-k 5
  ```
- Trigger classification immediately after summarisation with the built-in flag (skips when `--ccs-id/--ccs-path` are supplied):
  ```bash
  OPENAI_API_KEY=... \
  python Pre-Processing/summarize_pdf.py thesis/WISS/2024/001_parasights-2.pdf \
      --extractor grobid \
      --classify-ccs \
      --output summaries/wiss2024_001.json
  ```
- Classify existing summary JSON files (up to 3 labels per paper by default). The script narrows candidates via Gemini embeddings before delegating the final choice to the OpenAI Responses API:
  ```bash
  OPENAI_API_KEY=... \
  python Pre-Processing/ccs/classify_ccs.py summaries/*.json \
      --model gpt-5 \
      --embedding-model gemini-embedding-001 \
      --top-candidates 15 \
      --max-concepts 3 \
      --output classification_report.jsonl
  ```
  Use `--embedding-model none` if you prefer keyword filtering instead of embeddings, and add `--update` to patch the input JSON files in-place (fields `ccs.ids`, `ccs.paths`, and `ccs.llm_explanations` are populated).
- Both CLIs assume the ACM CCS XML lives at `ACM CCS/acm_ccs2012-1626988337597.xml`; override the location with `--xml` when needed.

### 4.7 Batch embedding refresher (new)
- Once summaries exist you can regenerate or backfill embeddings with the dedicated CLI. It supports Gemini, Vertex AI and local Sentence Transformer backends, adds metadata (`embedding_meta`) and honours per-section filters. The Gemini path deduplicates identical section text, prefers `batch_embed_content` for throughput, and logs a warning before falling back to sequential requests if batching is unavailable:
  ```bash
  GEMINI_API_KEY=... \
    python Pre-Processing/compute_embeddings.py summaries/*.json \
        --provider gemini \
        --model gemini-embedding-001 \
      --embedding-version v1 \
      --normalize
  ```
  - Add `--force` to overwrite existing vectors. プロバイダやモデルを切り替える際も `--force` を付けて再計算してください。
  - Use `--output-dir refreshed/` to keep originals untouched; the original subdirectory layout (e.g. `summaries/2024/001.json`) is preserved under the output directory.
  - Switch backends with `--provider vertex-ai --model text-embedding-005` (remember `--vertex-project` / `--vertex-location`) or `--provider local --model intfloat/multilingual-e5-large-instruct`.
  - `--sections positioning purpose` restricts embeddings to certain summary fields when needed.

### 5. Troubleshooting
- `GROBID service not reachable`: run the container, expose it on a reachable IP, increase `--grobid-timeout`.
- `requests package missing`: install inside the venv (`pip install requests`).
- `pypdf import error`: install `pypdf` or `PyPDF2`.
- Large PDFs causing timeouts: raise `--grobid-timeout` and consider `docker run` with `-m` to allocate more RAM.

### 6. Known gaps / roadmap
- `call_openai` still aborts on repeated HTTP 429/5xx responses; an exponential backoff + jitter retry layer is planned but not yet implemented.
- Structured job logging (`processing_jobs` table, per-request token accounting) is deferred; check stderr output for now when diagnosing failures.
- Embedding caches are in-memory only—consider persisting section hashes if you process very large corpora and want to skip recomputation between runs.

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
- Vertex AI を使う場合は `.env` に `VERTEX_AI_PROJECT` / `VERTEX_AI_LOCATION` / `VERTEX_AI_EMBEDDING_MODEL`（必要なら `VERTEX_AI_EMBEDDING_DIM`）を追加し、Google Cloud の認証情報を環境変数や gcloud auth で用意してください。
- 任意: GROBID を動かす Docker (メモリ 4GB 以上を推奨)。

### 1. WISS PDF の取得
- まだ proceedings の CSV が無い場合は `Pre-Processing/scraipingpdf/` にあるスクレイパーを利用します。最近のページは次のコマンドで取得できます:
  ```bash
  .venv/bin/python Pre-Processing/scraipingpdf/scraiping2.py \
      https://www.wiss.org/WISS2024Proceedings/ \
      -o WISSProceedings/wiss2024.csv
  ```
  古いレイアウト（2004〜2020 年頃）は `scraiping.py` に複数 URL や URL 一覧ファイル（1 行 1 URL）を与えて対応します:
  ```bash
  .venv/bin/python Pre-Processing/scraipingpdf/scraiping.py \
      -f urls.txt \
      -o WISSProceedings/wiss2004-2009.csv
  ```
  必要に応じて `pip install requests beautifulsoup4 lxml` を仮想環境にインストールしてください。出力 CSV には `id`, `title`, `authors` に加え、PDF / 動画 / 査読コメントなどの絶対 URL が含まれます。
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
  - `--max-workers`: レートリミッター付きで並列ダウンロードする場合に設定（許可がある場合のみ増やしてください）。
  - `--overwrite`: 既存ファイルを上書き。

### 1.5 スクレイピング時の注意
- WISS の利用規約を遵守し、権限のある資料のみ取得してください。
- デフォルトの 1 秒ディレイ（またはそれ以上）を守り、`--years` で対象を絞り込むと負荷を抑えられます。
- 取得した PDF はローカルに保存し、再要約時は再ダウンロードせずキャッシュを活用してください。
- HTTP エラー（4xx/5xx）が続く場合は処理を中断し、原因を確認しましょう。
- クロール前に `robots.txt` や公開ポリシーを確認し、アクセス禁止の指示がある場合は従ってください。
- ダウンローダーを改造する際は `urllib.request.Request` などで丁寧な User-Agent を付与し、並列化する場合も既定値（1 ワーカー）を基準に、ホストから明示的な許可があるケースだけ `--max-workers` を増やしてください。
- 有料ページや認証が必要な領域を回避したり回避策を講じたりしないでください。判断が難しい場合は必ず許可を取り、取得した資料の扱い（共有可否など）にも注意してください。

### 2. GROBID の起動（推奨）
- [GROBID](https://grobid.readthedocs.io/) は学術 PDF を TEI XML 構造に変換するオープンソースの解析サービスで、PyPDF より高品質な本文テキストとメタデータを取得できます。
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
- GROBID を使わない場合は `--extractor pypdf` を指定します（事前に `pypdf` を導入）。PyPDF ルートでは簡易ヘッダ検出で疑似セクションを付与しているため、後続のチャンク処理やサマリ生成は継続できますが、ヘッダ品質は GROBID より劣る可能性があります。
- 主なオプション:
  - `--title`, `--author`, `--year`, `--pdf-link`, `--code-link`
  - `--ccs-path`, `--ccs-id`（複数指定可）
  - `--chunk-size` / `--overlap`（既定値 2500 / 250 文字）
  - `--temperature`（サンプリング温度の調整。GPT-5 ファミリーを選ぶと無視されます）
  - `--no-dual-language`：英語フィールドを生成せず日本語のみ出力したい場合
  - `--no-flatten-translations`：英語フィールドを `translations.en` のみに残し、トップレベルの `*_en` キーを作成しない
  - `--embeddings --embedding-provider local`（Sentence Transformers でサマリー各セクション：位置づけ／目的／手法／評価（可能なら概要）を個別に埋め込み化）
  - `--embeddings --embedding-provider vertex-ai --embedding-model text-embedding-004`（同じセクション構造で Vertex AI Embeddings を利用）

### 3.5 要約の方針
- 整形した本文テキストを OpenAI Responses API に投げ、「位置づけ」「目的」「手法」「評価」の 4 セクションで返すようプロンプトを厳密に指定しています。
- 各セクションは 2 文（最大 3 文）を目安とし、多数の論文を比較しやすい短く筋の通ったまとめを意識しています。
- `--language` フラグで見出しと言語を日本語・英語で切り替えつつ、JSON のキーは共通なので後続処理の互換性が保たれます。
- 最終出力は既定で日英両方のフィールド（`title` / `title_en` など）を含みます。英語を除外したい場合は `--no-dual-language`、トップレベルへのコピーを避けたい場合は `--no-flatten-translations` を併用してください。
- 日本語出力時の見出しは `概要`、`研究の位置づけ／目的`、`ソリューション／エンジニアリング`、`評価` で、スクリプト内の `JAPANESE_HEADINGS` に対応しています。
- 元資料に評価指標・データセット・参加人数などが記されていればそのまま記載し、欠けている場合は `"Not specified"` / `記載なし` で明示します。
- セクション検出は GROBID が返す構造を優先し、利用できない場合は PyPDF で抽出した段落を分割して補完します。

### 3.6 バッチオーケストレーター（新機能）
- 抽出 → 要約 → 埋め込み → ACM CCS 分類までを一括実行する CLI です。軽量な SQLite キューでリトライや再開を管理します:
  ```bash
  .venv/bin/python Pre-Processing/orchestrator.py run \
      --pdf-dir thesis/WISS/2024 \
      --output-dir summaries/wiss2024 \
      --env-file .env \
      --pretty
  ```
- 既定では `*.pdf` を再帰的に探索し、出力ディレクトリ配下に元の階層構造を保った JSON（例: `summaries/wiss2024/001_parasights-2.json`）を生成します。完了済みジョブを再処理したいときは `--force` を付けてください。
- サマライザ用の設定を JSON でまとめておくと再現性が高まります。例:
  ```json
  {
    "model": "gpt-5-mini",
    "language": "Japanese",
    "chunk_size": 2800,
    "embedding_provider": "gemini",
    "gemini_embedding_model": "gemini-embedding-001",
    "classify_ccs": true
  }
  ```
  このファイルを `--config configs/pipeline_wiss2024.json` のように指定し、追加で `--model` や `--language`、`--embedding-provider`、`--disable-embeddings`、`--disable-ccs` などの CLI フラグを上書き用途で利用します。
- 進捗やエラー件数は次のコマンドで確認できます:
  ```bash
  python Pre-Processing/orchestrator.py status --db data/pipeline_state.db
  ```

### 4. 出力 JSON
- 生成される JSON のイメージ:
  ```jsonc
  {
    "id": "WISS2024-001",
    "title": "ParaSights：…",               // 日本語
    "title_en": "ParaSights: ...",        // 英語（既定で生成）
    "authors": ["..."],
    "authors_en": ["..."],
    "abstract": "...",
    "abstract_en": "...",
    "positioning_summary": "...",
    "positioning_summary_en": "...",
    "purpose_summary": "...",
    "purpose_summary_en": "...",
    "method_summary": "...",
    "method_summary_en": "...",
    "evaluation_summary": "...",
    "evaluation_summary_en": "...",
    "year": 2024,
    "ccs": {"paths": ["..."], "ids": ["..."]},
    "links": {"pdf": "...", "code": "..."},
    "metadata_meta": {
      "doi": {
        "status": "consistent",
        "confidence": "high",
        "preferred_source": "grobid",
        "sources": {"grobid": "...", "metadata_llm": "...", "synthesized": "...", "cli": null},
        "final": "..."
      },
      "year": {
        "status": "consistent",
        "confidence": "medium",
        "preferred_source": "metadata_llm",
        "sources": {"grobid": 2024, "metadata_llm": 2024, "synthesized": 2024, "cli": null},
        "final": 2024
      }
    },
    "translations": {
      "en": {
        "title": "ParaSights: ...",
        "abstract": "...",
        "positioning_summary": "...",
        "purpose_summary": "...",
        "method_summary": "...",
        "evaluation_summary": "...",
        "links": {"pdf": "...", "code": "..."}
      }
    },
    "embeddings": {
      "provider": "...",
      "model": "...",
      "dim": 768,
      "normed": true,
      "sections": {
        "positioning": [...],
        "purpose": [...],
        "method": [...],
        "evaluation": [...],
        "abstract": [...]
      }
    }
  }
  ```
- 各 summary フィールドは 2 文（最大 3 文）で、課題・目的・手法・評価＋結果を素早く把握できるようにしています。

### 4.5 セマンティック検索（任意）
- 生成済み要約のディレクトリから、セクション単位の埋め込み索引を作成:
  ```bash
  .venv/bin/python Pre-Processing/search/build_embedding_index.py \
      summaries/*.json \
      -o search/embedding_index.jsonl
  ```
  文字数が短すぎるセクション（既定で 40 文字未満）は除外し、埋め込みベクトルとメタ情報を JSONL に蓄積します。
- フリーテキストで索引を検索。複数のプロバイダ／モデルが混在する場合は引数で指定してください:
  ```bash
  .venv/bin/python Pre-Processing/search/search_embeddings.py \
      search/embedding_index.jsonl \
      --query "高齢者のための共有型AR体験" \
      --top-k 5 \
      --provider vertex-ai \
      --vertex-project あなたのGCPプロジェクトID \
      --vertex-location us-central1
  ```
- `--vertex-project` / `--vertex-location` を省略すると、`VERTEX_AI_PROJECT`（もしくは `GOOGLE_CLOUD_PROJECT`）と `VERTEX_AI_LOCATION` の環境変数が利用されます（既定ロケーションは `us-central1`）。
- ローカル（Sentence Transformer）で埋め込みを生成した場合は Vertex 関連のフラグを省略すれば同じモデルが再利用されます。
- 既存の要約 JSON を照会として使いたい場合は `--query-json` に指定すると、位置づけ／目的／手法／評価／概要の各セクションごとに類似文書が表示されます。
- `--exclude-paper 論文ID` を付ければ特定の論文を結果から除外できます（要約 JSON を指定した場合、その `id` は自動的に除外されます）。
- JSONL 索引はダッシュボードや外部ベクターストア（Pinecone, Weaviate など）の入力として再利用でき、要約処理を繰り返す必要がありません。

### 4.6 ACM CCS 分類（LLM 活用）
- 要約生成と同時に分類したい場合は `--classify-ccs` を付けて実行します（`--ccs-id` / `--ccs-path` を指定した場合は自動分類をスキップ）:
  ```bash
  OPENAI_API_KEY=... \
  python Pre-Processing/summarize_pdf.py thesis/WISS/2024/001_parasights-2.pdf \
      --extractor grobid \
      --classify-ccs \
      --output summaries/wiss2024_001.json
  ```
- 既存の要約 JSON を対象にする場合は専用 CLI を使います。Gemini 埋め込みで候補を絞り込んだ上で OpenAI Responses API に 3 件までの最適な概念を選ばせます:
  ```bash
  OPENAI_API_KEY=... \
  python Pre-Processing/ccs/classify_ccs.py summaries/*.json \
      --model gpt-5 \
      --embedding-model gemini-embedding-001 \
      --top-candidates 15 \
      --max-concepts 3 \
      --output classification_report.jsonl
  ```
- タクソノミーの辞書やキーワード探索には `Pre-Processing/ccs/export_taxonomy.py` を利用できます。既定ではリポジトリ内の `ACM CCS/acm_ccs2012-1626988337597.xml` を参照しますが、`--xml` で明示的に切り替えられます。

### 4.7 埋め込み再計算バッチ（新機能）
- 要約 JSON が揃っている場合、専用 CLI でセクション別埋め込みを再生成できます。Gemini / Vertex AI / Sentence Transformers に対応し、`embedding_meta` にメタ情報を追記します。Gemini 経路では同一テキストをまとめて `batch_embed_content` に投げ、利用できない場合は警告を出して逐次リクエストにフォールバックします:
  ```bash
  GEMINI_API_KEY=... \
    python Pre-Processing/compute_embeddings.py summaries/*.json \
        --provider gemini \
        --model gemini-embedding-001 \
      --embedding-version v1 \
      --normalize
  ```
- 主な使い方:
  - `--force`: 既存ベクトルの上書き。プロバイダやモデルを切り替える際は付与してください。
  - `--output-dir refreshed/`: 元の JSON を残したまま別ディレクトリに書き出します（サブディレクトリ構造は保持）。
  - `--provider vertex-ai --model text-embedding-005`: Vertex AI へ切り替える例（`--vertex-project` / `--vertex-location` を忘れず指定）。
  - `--provider local --model intfloat/multilingual-e5-large-instruct`: ローカル Sentence Transformer を利用する例。
  - `--sections positioning purpose`: 特定セクションだけ埋め込みたい場合に列挙。

### 5. よくあるトラブル
- `GROBID service not reachable`: Docker コンテナが起動しているか、IP/ポートがアクセス可能か、`--grobid-timeout` が短すぎないか確認。
- `requests が見つからない`: 仮想環境内で `pip install requests` を実行。
- `pypdf が見つからない`: `pip install pypdf` もしくは `PyPDF2` を導入。
- 大きな PDF でタイムアウトする: `--grobid-timeout` を増やし、必要なら `docker run` でメモリ上限 (`-m`) を引き上げてください。

### 6. 既知の課題 / 今後の対応
- `call_openai` はまだ HTTP 429 / 5xx 用の指数バックオフやジッタを備えていません。リトライ層とトークン使用量の記録は今後の実装予定です。
- ログは標準エラー出力中心で、`processing_jobs` のような永続ジョブログは未整備です。大規模処理では追加の監視・記録を検討してください。
- 埋め込みの再利用キャッシュはプロセス内のみです。巨大なコーパスで重複計算を避けたい場合は、セクション本文のハッシュを保存する仕組みを追補する必要があります。

エンジニア視点でパイプライン全体を把握できるように設計しているので、構成やフラグを調整しつつ、WISS 論文の比較・整理に活用してください。
