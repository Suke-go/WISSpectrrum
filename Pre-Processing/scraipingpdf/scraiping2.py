#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汎用版: WISS 2024/2025など、<div class="entry"> と <span class="title"> 両対応
"""

import requests
from bs4 import BeautifulSoup
import csv, sys, time, re
from urllib.parse import urljoin

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def fetch(url):
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        r.raise_for_status()
        if not r.encoding or r.encoding.lower() in ("iso-8859-1", "us-ascii"):
            r.encoding = r.apparent_encoding
        return r.text
    except Exception as e:
        print(f"[WARN] fetch失敗: {url} ({e})", file=sys.stderr)
        return None

def parse_entry_type1(div, base_url):
    """<div class='entry'> 型"""
    title_p = div.select_one("p.title")
    raw_title = title_p.get_text(" ", strip=True) if title_p else ""
    m = re.match(r"\s*\[(\d+)\]\s*(.*)", raw_title)
    paper_id = m.group(1) if m else ""
    title = m.group(2).strip() if m else raw_title.strip()
    authors = div.select_one("p.authors")
    authors_text = authors.get_text(" ", strip=True) if authors else ""

    pdf = mp4 = review = ""
    for a in div.select("a[href]"):
        href = urljoin(base_url, a["href"])
        txt = a.get_text(strip=True)
        if "pdf" in txt.lower() or href.lower().endswith(".pdf"):
            pdf = href
        elif "mp4" in txt.lower() or "video" in txt.lower():
            mp4 = href
        elif "査読" in txt or "review" in txt.lower():
            review = href

    return {"id": paper_id, "title": title, "authors": authors_text,
            "pdf_url": pdf, "mp4_url": mp4, "review_url": review,
            "source_page": base_url}

def parse_entry_type2(p, base_url):
    """<p><span class='title'> 型"""
    title_tag = p.select_one("span.title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    pdf = mp4 = review = ""
    for a in p.select("a[href]"):
        href = urljoin(base_url, a["href"])
        txt = a.get_text(strip=True)
        if "pdf" in txt.lower() or href.lower().endswith(".pdf"):
            pdf = href
        elif "mp4" in txt.lower() or "video" in txt.lower() or "デモ" in txt:
            mp4 = href
        elif "査読" in txt or "review" in txt.lower():
            review = href
    authors_tag = p.select_one("span.authors")
    authors = authors_tag.get_text(" ", strip=True) if authors_tag else ""
    return {"id": "", "title": title, "authors": authors,
            "pdf_url": pdf, "mp4_url": mp4, "review_url": review,
            "source_page": base_url}

def parse_page(html, url):
    soup = BeautifulSoup(html, "lxml")
    entries = []
    for div in soup.select("div.entry"):
        entries.append(parse_entry_type1(div, url))
    for p in soup.select("p"):
        if p.select_one("span.title") and p.select_one("span.authors"):
            entries.append(parse_entry_type2(p, url))
    return entries

def main():
    if len(sys.argv) < 3:
        print("使い方: python scrape_entries_to_csv.py <URL> -o <出力CSV>", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]
    if "-o" not in sys.argv:
        print("出力ファイル指定(-o)が必要です。", file=sys.stderr)
        sys.exit(1)
    output = sys.argv[sys.argv.index("-o") + 1]

    html = fetch(url)
    if not html:
        sys.exit(1)
    rows = parse_page(html, url)
    print(f"{len(rows)} entries found")

    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "authors", "pdf_url", "mp4_url", "review_url", "source_page"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"✅ CSV書き出し完了: {output}")

if __name__ == "__main__":
    main()
