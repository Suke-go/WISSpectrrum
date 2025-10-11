#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# THIS IS ONLY FROM 2020
"""
WISSなどの一覧ページから <div class="entry"> をスクレイピングして CSV を生成
使い方:
  python scraiping2.py https://www.wiss.org/WISS2004Proceedings/ -o wiss2004.csv
  python scrape_entries_to_csv.py -f urls.txt -o out.csv
必要:
  pip install requests beautifulsoup4 lxml
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

UA = "Mozilla/5.0 (compatible; EntryScraper/1.0; +https://example.local)"

def fetch(url: str, timeout: float = 20.0) -> Optional[str]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        r.raise_for_status()
        # 文字化け回避: サーバーの宣言 > apparent_encoding の順で決める
        if not r.encoding or r.encoding.lower() in ("iso-8859-1", "us-ascii"):
            r.encoding = r.apparent_encoding or "utf-8"
        return r.text
    except requests.RequestException as e:
        print(f"[WARN] fetch失敗: {url} ({e})", file=sys.stderr)
        return None

def text_without_links(tag: Tag) -> str:
    # タイトル<p class="title">から<a>を除いた素のテキストを抽出
    t = tag.__copy__() if hasattr(tag, "__copy__") else tag
    # copyがない環境向け: 子aを一時的に取り除いてテキスト抽出
    parts = []
    for child in tag.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag) and child.name != "a":
            parts.append(child.get_text(" ", strip=False))
        else:
            # <a> は無視
            pass
    return " ".join(parts).strip()

def parse_entry_div(entry: Tag, base_url: str) -> Dict[str, str]:
    # タイトル行
    title_p = entry.select_one("p.title")
    raw_title = text_without_links(title_p) if title_p else ""
    # 例: "[12] ボタンひとつで..." → id と本題に分割
    m = re.match(r"\s*\[(\d+)\]\s*(.*)", raw_title)
    paper_id = m.group(1) if m else ""
    title = m.group(2).strip() if m else raw_title.strip()

    # 著者
    authors_p = entry.select_one("p.authors")
    authors = authors_p.get_text(" ", strip=True) if authors_p else ""

    # リンク群
    pdf_url = mp4_url = comments_url = ""
    other_links = []
    for a in entry.select("a[href]"):
        href = urljoin(base_url, a["href"])
        label = a.get_text(strip=True).lower()
        if href.lower().endswith(".pdf") or "pdf" in label:
            pdf_url = href
        elif href.lower().endswith(".mp4") or "mp4" in label or "video" in label:
            mp4_url = href
        elif href.lower().endswith(".html") or "査読" in label or "comment" in label:
            comments_url = href
        else:
            other_links.append(href)

    return {
        "id": paper_id,
        "title": title,
        "authors": authors,
        "pdf_url": pdf_url,
        "mp4_url": mp4_url,
        "comments_url": comments_url,
        "source_page": base_url,
        "other_links": " ".join(other_links),
    }

def parse_page(html: str, url: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    entries = soup.select("div.entry")
    if not entries:
        # 予備: 構造が少し違うページ向けに近い構造も探索
        entries = soup.select("div.paper, div.item, li.entry")
    rows = [parse_entry_div(e, url) for e in entries]
    return rows

def load_urls_from_file(path: Path) -> List[str]:
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        urls.append(s)
    return urls

def main():
    ap = argparse.ArgumentParser(description="エントリ一覧をスクレイピングしてCSV出力")
    ap.add_argument("urls", nargs="*", help="スクレイピングするURL（複数可）")
    ap.add_argument("-f", "--from-file", type=str, help="URLを1行ずつ書いたテキストファイル")
    ap.add_argument("-o", "--output", type=str, required=True, help="出力CSVパス")
    ap.add_argument("--sleep", type=float, default=0.5, help="アクセス間スリープ秒（礼儀）")
    ap.add_argument("--encoding", type=str, default="utf-8-sig", help="CSV文字コード（Excel向けはutf-8-sig推奨）")
    args = ap.parse_args()

    urls: List[str] = list(args.urls)
    if args.from_file:
        urls += load_urls_from_file(Path(args.from_file))
    if not urls:
        print("URLが指定されていません。引数または --from-file で指定してください。", file=sys.stderr)
        sys.exit(1)

    all_rows: List[Dict[str, str]] = []
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] GET {url}", file=sys.stderr)
        html = fetch(url)
        if html is None:
            continue
        rows = parse_page(html, url)
        print(f"  -> {len(rows)} entries", file=sys.stderr)
        all_rows.extend(rows)
        time.sleep(args.sleep)

    # 重複（id+title）でユニーク化（同じページを複数回入れたときの保険）
    seen = set()
    uniq_rows = []
    for r in all_rows:
        key = (r.get("id", ""), r.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        uniq_rows.append(r)

    fieldnames = [
        "id", "title", "authors",
        "pdf_url", "mp4_url", "comments_url",
        "source_page", "other_links"
    ]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding=args.encoding) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in uniq_rows:
            writer.writerow(r)

    print(f"CSVを書き出しました: {out_path}  ({len(uniq_rows)} rows)", file=sys.stderr)

if __name__ == "__main__":
    main()
