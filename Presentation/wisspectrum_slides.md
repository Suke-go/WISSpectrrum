---
marp: true
paginate: true
footer: "WISSpectrrum | 清水紘輔"
style: |
  :root {
    color: #f0f0f0;
    background-color: #0b0b0b;
  }
  section {
    font-family: "Noto Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-weight: 300;
    letter-spacing: 0.02em;
  }
  h1, h2, h3 {
    font-weight: 500;
    letter-spacing: 0.05em;
  }
  strong {
    color: #6ec1ff;
  }
  a {
    color: #6ec1ff;
  }
  ul li {
    margin-bottom: 0.4rem;
  }
---

<!-- _class: lead -->

# WISSpectrrum  
## WISS論文の高速サマリーワークフロー

Kosuke Shimizu,
Colleage of Media Arts, Science and Technology, 
University of Tsukuba
shimizu@ai.iit.tsukuba.ac.jp

---

## 目次

- 目的と背景
- ワークフローの全体像
- 主要コンポーネント
- 成果と活用シーン
- 今後の展望

---

## 目的と背景

- WISS論文を大量に収集・要約し、研究動向を迅速に把握したい  
- 手動での情報抽出は時間がかかり、再現性も低い  
- 自動化されたパイプラインで、収集から要約までを一貫処理

---

## ワークフローの全体像

1. Proceedingsサイトから論文リストを取得  
2. PDFを自動ダウンロードし、ディレクトリを整理  
3. GROBIDなどでテキスト抽出  
4. 大規模言語モデルで要約を生成  
5. JSON形式で構造化し、検索や分析へ活用

---

## 主要コンポーネント

- **スクレイピング**: `scraipingpdf/` が年次ごとのProceedingsを解析  
- **ダウンロード**: `pdfdownloader/` が高速・安全にPDFを収集  
- **要約スクリプト**: `summarize_pdf.py` が抽出・要約・埋め込みを一括実行  
- **柔軟性**: GROBIDとPyPDFを切替可能、言語やチャンクサイズも可変

---

## 成果と活用シーン

- 4つのセクション（Positioning・Purpose・Method・Evaluation）で情報を整理  
- JSON出力でダッシュボード連携や検索システムへの統合が容易  
- 埋め込み生成により、類似研究探索や推薦にも応用可能

---

## 今後の展望

- ダウンロードから要約までの完全自動化と監視  
- 評価指標やデータセット情報の深掘り抽出  
- 共有ダッシュボードやアラート機能による研究支援

---


