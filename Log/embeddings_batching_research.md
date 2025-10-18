# Gemini Embedding 最適化メモ

## 対応方針
- `google.generativeai.batch_embed_content` が利用可能な場合はバッチ API を優先し、未対応環境では順次フォールバック。
- テキスト本文で重複するセクションは事前にまとめ、1 回の API 呼び出し結果を複数キーで共有する。
- エラー発生時は警告を出して従来の逐次処理へ切り替える。

## 参考にしたい検索クエリ
1. `site:cloud.google.com "batch_embed_content" google generative ai`
2. `google generative ai python embeddings batch request`
3. `gemini embed_content API examples python`

## 保留課題
- バッチ API のレスポンス仕様はバージョンによって差異があるため、最新ドキュメントで JSON 構造を要確認。
- レートリミット超過時の指数バックオフとリトライ戦略は `call_openai` と同様の仕組みを導入する。
- 生成された埋め込みをローカルキャッシュ（ハッシュキー）に保存し、同一テキストの重複計算をさらに削減する。

