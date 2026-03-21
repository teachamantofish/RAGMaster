
"""Standalone web crawler test harness (CLI only, not wired to UI).

Workflow:
1) Read URLs from a text file (one URL per line).
2) Fetch each page, parse/clean with BeautifulSoup using include/exclude selectors.
3) Convert cleaned HTML to Markdown with markdownify.

Run from repo root:
	python run_python_entry.py Get_and_Chunk/2crawlwebtest.py
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urldefrag, urlparse

from bs4 import BeautifulSoup
from markdownify import markdownify as md
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


##################################################

URL_LIST_FILE = Path("C:\\GIT\\Z_Master_Rag\\Data\\crawlurls.md")
OUTPUT_DIR = Path("C:\\GIT\\Z_Master_Rag\\Data\\_crawl_test")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

def get_urls(path):
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("http"):
            urls.append(line)
    return urls

def fetch_html(url):
    r = session.get(url, timeout=20)
    r.raise_for_status()
    return r.text

urls = get_urls(URL_LIST_FILE)

for i, url in enumerate(urls, 1):
    try:
        html = fetch_html(url)
        out = OUTPUT_DIR / f"page_{i}.html"
        out.write_text(html, encoding="utf-8")
        print(f"saved: {url}")
    except Exception as e:
        print(f"failed: {url} ({e})")
		
#############################################################

