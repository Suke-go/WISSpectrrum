# 実装進捗サマリー

## 現状概要
- PDF スクレイピングから要約生成・埋め込み計算・CCS 分類までの CLI スクリプト群が整備され、ローカルで一連の前処理パイプラインを実行できる状態。
- ログ永続化や DB 連携、`processing_jobs` テーブルといった運用メタデータの管理機構は未実装で、手動実行前提のワークフロー。
- 類似度可視化 UI や探索体験のフロントエンドは未着手。現在は JSON/JSONL 生成と CLI ベースの検索のみ。

## 工程別の進捗

### 1. PDF 収集・管理
- `Pre-Processing/scraipingpdf/scraiping.py`, `scraiping2.py` で過去/最新フォーマット双方の WISS サイトからエントリ一覧を CSV 化可能。
- `Pre-Processing/pdfdownloader/download_wiss_pdfs.py` で CSV から年別ディレクトリへ PDF を保存（レートリミット・並列 DL・失敗レポート対応）。
- メタデータは CSV ベースで保持。Master ドキュメントに触れられている専用 DB / catalog 連携はまだ無い。

### 2. テキスト抽出
- `Pre-Processing/summary/pdf_extractors.py` に PyPDF フォールバックと GROBID 連携を実装。TEI ベースのセクション抽出とフォールバック処理あり。
- GROBID の起動スクリプトや Docker 設定は README 参照レベルで、稼働監視・再試行キューなどは未着手。

### 3. セクション別要約生成
- `Pre-Processing/summary/summarize_pdf.py` がチャンク分割、断片要約、最終 JSON 合成までを自動化。OpenAI Responses API を使用し、欠損管理・少量の再試行ロジックを内包。
- 出力には `positioning/purpose/method/evaluation` 各サマリー、メタデータ、任意で埋め込みベクトルを含められる。
- Master ドキュメントで言及されていた `processing_jobs` テーブルやエラーログの永続化、部分再試行の細分化はまだ組み込まれていない。

### 4. 埋め込み計算
- `Pre-Processing/embeddings.py` がローカル (`sentence-transformers`) / Vertex AI 双方のバックエンドをカプセル化。`_lazy_*` キャッシュとスレッドロックで再利用効率を確保しつつ、正規化（L2）有無も指定可能。
- Summariser (`Pre-Processing/summary/summarize_pdf.py`) の `--embeddings` フラグで各セクション（positioning/purpose/method/evaluation/abstract）のベクトルを生成し、JSON 出力へ `embeddings.sections` とメタ情報（provider/model/dim/normed）を付加。
- Gemini Embedding API を呼び出すヘルパーを追加し、`Pre-Processing/compute_embeddings.py` CLI で要約 JSON 群に対するバッチ埋め込み計算（Gemini / Vertex AI / Sentence Transformers 切り替え、再計算・バージョン管理・メタ情報付与）をサポート。
- 既存ベクトルを上書きする際は `--force` 指定が必須。`--output-dir` を指定すると元のサブディレクトリ構造（例: `summaries/2024/001.json`）を保ったまま別ツリーへ書き出す。
- 生成ベクトルはファイル内保存に留まり、pgvector 等の DB 連携・永続化やジョブ管理・エラーログ統合はこれから。バッチ処理・再計算時のリジューム機構も未整備。

### 5. 類似度計算・検索
- `Pre-Processing/search/build_embedding_index.py` が要約 JSON 群からセクション別ベクトルを抽出し JSONL インデックス化。
- `Pre-Processing/search/search_embeddings.py` でクエリの埋め込み生成とコサイン類似度ランキングを提供。CLI 出力のみで、UI 可視化は未着手。

### 6. ACM CCS 分類
- `Pre-Processing/ccs/` 配下にタクソノミーのパース (`taxonomy.py`)、埋め込みを用いた候補抽出 (`classifier.py`)、LLM を使った分類 CLI (`classify_ccs.py`) を実装。
- Summariser からの自動分類起動や分類結果の保存先統合は設定されておらず、別 CLI での後処理運用前提。

## DB 管理改善計画
- `.env` と Streamlit UI に論文ルート（例 `WISS_DATA_ROOT`）を追加し、`orchestrator` が相対パスでジョブ登録できるよう整備する。既存レコードを基準ルート＋相対パスへ変換するメンテ CLI を用意する。
- `STATE_DB_PATH` の既定値をリポジトリ直下からユーザローカル（例 `%LOCALAPPDATA%/WISS/pipeline_state.db`）へ移し、PC 間コピーでジョブが混在しないようにする。切り替え手順としてバックアップ・移行ガイドを整備する。
- `jobs` テーブルに `base_root` と `relative_pdf` 列を追加し、新規ジョブはそこで管理する。旧 `pdf_path` 列は互換期間中残し、整合チェックと変換状況を出力するスクリプトを添える。
- Streamlit と CLI の両方に DB パス選択機能を追加し、意図しない旧 DB 参照時は警告を表示する。
- 失敗ジョブや旧 PC のジョブを整理する掃除コマンド（削除・VACUUM・アーカイブ）を `PipelineState` 周りに実装し、定期運用手順をドキュメント化する。
## 既知の未実装/課題
- UI・可視化レイヤ（ネットワーク図・調整 UI など）は未開始。
- バッチ実行やリトライ制御、ジョブ管理まわりは設計のみでコード化されていない。
- ローカル JSON/JSONL のまま管理しており、DB 連携・pgvector などの永続層整備は未着手。
- コスト/品質ログ、トークン使用量の記録、自動テスト・CI などの運用基盤も未整備。
