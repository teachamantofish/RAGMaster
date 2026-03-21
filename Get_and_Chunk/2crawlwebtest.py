
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


# =============================
# Configuration Constants
# =============================

# Input and output locations
URL_LIST_FILE = Path("Data/crawlurls.md")
OUTPUT_DIR = Path("Data/_crawl_test")

# URL list parsing
SKIP_EMPTY_LINES = True
SKIP_COMMENT_LINES = True
COMMENT_PREFIX = "#"
REQUIRE_HTTP_URLS = True
APPEND_HTML_IF_MISSING = False

# Crawl loop control
MAX_URLS: int | None = 1
STOP_ON_FIRST_ERROR = False
OVERWRITE_EXISTING_FILES = True
SKIP_IF_OUTPUT_EXISTS = True
CONCURRENT_WORKERS = 1

# Request behavior
USER_AGENT = "Mozilla/5.0 (compatible; Z_Master_Rag-TestCrawler/1.0)"
REQUEST_TIMEOUT_SEC = 4
REQUEST_CONNECT_TIMEOUT_SEC = 1
REQUEST_VERIFY_SSL = True
REQUEST_RETRY_COUNT = 1
REQUEST_BACKOFF_SECONDS = 0
REQUEST_RETRY_STATUS_CODES = [429, 500, 502, 503, 504]
USE_ENV_PROXY = True

# BeautifulSoup cleaning behavior
INCLUDE_SELECTORS = [
	".position",
]
REQUIRE_INCLUDE_MATCH = False
EXCLUDE_SELECTORS = [
	".language-Navigation",
	".dexter-Author-Hide",
]
REMOVE_TAGS = ["script", "style", "noscript", "svg", "iframe", "nav", "footer", "header", "form"] # Remove entire elements of these types

# Markdown generation
MARKDOWNIFY_OPTIONS = {
	"heading_style": "ATX",
	"bullets": "-",
	"strip": ["img"],
}
MIN_MARKDOWN_CHARS = 100
WRITE_SOURCE_URL_HEADER = True


REPO_ROOT = Path(__file__).resolve().parent.parent

##################################################

URL_LIST_FILE = Path("Data/crawlurls.md")
OUTPUT_DIR = Path("Data/_crawl_test")

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



def build_http_session() -> requests.Session:
	if REQUEST_RETRY_COUNT <= 1:
		adapter = HTTPAdapter(max_retries=0)
	else:
		retry = Retry(
			total=REQUEST_RETRY_COUNT,
			connect=REQUEST_RETRY_COUNT,
			read=REQUEST_RETRY_COUNT,
			status=REQUEST_RETRY_COUNT,
			backoff_factor=REQUEST_BACKOFF_SECONDS,
			status_forcelist=REQUEST_RETRY_STATUS_CODES,
			allowed_methods=frozenset(["GET"]),
			raise_on_status=False,
		)
		adapter = HTTPAdapter(max_retries=retry)
	session = requests.Session()
	session.trust_env = USE_ENV_PROXY
	session.headers.update({"User-Agent": USER_AGENT})
	session.mount("http://", adapter)
	session.mount("https://", adapter)
	return session


def resolve_repo_path(path_value: Path) -> Path:
	path = Path(path_value)
	if path.is_absolute():
		return path
	return (REPO_ROOT / path).resolve()


def parse_url_list(url_list_path: Path) -> list[str]:
	if not url_list_path.exists():
		raise FileNotFoundError(f"URL list file does not exist: {url_list_path}")

	urls: list[str] = []
	for raw_line in url_list_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if SKIP_EMPTY_LINES and not line:
			continue
		if SKIP_COMMENT_LINES and line.startswith(COMMENT_PREFIX):
			continue

		if APPEND_HTML_IF_MISSING and not line.endswith(".html"):
			line = f"{line}.html"

		if REQUIRE_HTTP_URLS and not line.startswith(("http://", "https://")):
			raise ValueError(f"Non-http URL found in list: {line}")

		urls.append(line)

	if not urls:
		raise ValueError(f"URL list is empty after filtering: {url_list_path}")

	if MAX_URLS is not None:
		return urls[:MAX_URLS]
	return urls


def fetch_html(url: str, session: requests.Session) -> str:
	last_error: Exception | None = None
	for attempt in range(1, REQUEST_RETRY_COUNT + 1):
		try:
			response = session.get(
				url,
				timeout=(REQUEST_CONNECT_TIMEOUT_SEC, REQUEST_TIMEOUT_SEC),
				verify=REQUEST_VERIFY_SSL,
			)
			if response.status_code >= 400:
				raise RuntimeError(f"HTTP {response.status_code} for URL: {url}")
			return response.text
		except Exception as exc:
			last_error = exc
			if attempt >= REQUEST_RETRY_COUNT:
				break
			delay = REQUEST_BACKOFF_SECONDS * attempt
			print(f"  network error on attempt {attempt}/{REQUEST_RETRY_COUNT}; retrying in {delay}s")
			time.sleep(delay)

	assert last_error is not None
	raise RuntimeError(f"Failed after {REQUEST_RETRY_COUNT} attempts: {last_error}")


def clean_html(html: str, url: str) -> str:
	soup = BeautifulSoup(html, "html.parser")

	for tag_name in REMOVE_TAGS:
		for tag in soup.find_all(tag_name):
			tag.decompose()

	if INCLUDE_SELECTORS:
		included_nodes = []
		for selector in INCLUDE_SELECTORS:
			included_nodes.extend(soup.select(selector))

		if REQUIRE_INCLUDE_MATCH and not included_nodes:
			raise ValueError(f"No include selector matches for URL: {url}")

		if included_nodes:
			new_soup = BeautifulSoup("<div id='content-root'></div>", "html.parser")
			root = new_soup.select_one("#content-root")
			assert root is not None
			for node in included_nodes:
				root.append(node)
			soup = new_soup

	for selector in EXCLUDE_SELECTORS:
		for node in soup.select(selector):
			node.decompose()

	return str(soup)


def html_to_markdown(cleaned_html: str) -> str:
	markdown_text = md(cleaned_html, **MARKDOWNIFY_OPTIONS)
	markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text).strip()
	return markdown_text


def output_name_for_url(url: str) -> str:
	clean_url = urldefrag(url)[0]
	parsed = urlparse(clean_url)
	stem = Path(parsed.path).stem or "index"
	url_hash = hashlib.md5(clean_url.encode("utf-8")).hexdigest()[:8]
	return f"{stem}_{url_hash}.md"


def write_markdown(url: str, markdown_text: str, output_dir: Path) -> Path:
	if len(markdown_text) < MIN_MARKDOWN_CHARS:
		raise ValueError(f"Markdown too short ({len(markdown_text)} chars) for URL: {url}")

	output_dir.mkdir(parents=True, exist_ok=True)
	out_path = output_dir / output_name_for_url(url)

	if out_path.exists() and not OVERWRITE_EXISTING_FILES:
		raise FileExistsError(f"Output already exists and overwrite disabled: {out_path}")

	full_text = markdown_text
	if WRITE_SOURCE_URL_HEADER:
		full_text = f"Source URL: {url}\n\n{markdown_text}\n"

	out_path.write_text(full_text, encoding="utf-8", newline="\n")
	return out_path


def process_one_url(url: str, output_dir: Path, session: requests.Session) -> tuple[bool, str, str]:
	"""Return (success, url, message)."""
	out_path = output_dir / output_name_for_url(url)
	if SKIP_IF_OUTPUT_EXISTS and out_path.exists():
		return True, url, f"skip existing -> {out_path}"

	html = fetch_html(url, session)
	cleaned = clean_html(html, url)
	markdown_text = html_to_markdown(cleaned)
	final_path = write_markdown(url, markdown_text, output_dir)
	return True, url, f"saved -> {final_path}"


def main() -> int:
	url_list_path = resolve_repo_path(URL_LIST_FILE)
	output_dir = resolve_repo_path(OUTPUT_DIR)
	urls = parse_url_list(url_list_path)

	print(f"Loaded {len(urls)} URLs from {url_list_path}")
	print(f"Writing markdown output to {output_dir}")
	print(f"Workers: {CONCURRENT_WORKERS}")
	print(f"Use env proxy: {USE_ENV_PROXY}")

	failures: list[tuple[str, str]] = []
	saved_count = 0
	skipped_count = 0
	start_time = time.time()

	if CONCURRENT_WORKERS != 1:
		raise ValueError("CONCURRENT_WORKERS must be 1 for simple test mode")

	session = build_http_session()
	for idx, url in enumerate(urls, start=1):
		print(f"[{idx}/{len(urls)}] Fetching {url}")
		try:
			success, _, msg = process_one_url(url, output_dir, session)
			if msg.startswith("skip existing"):
				skipped_count += 1
			else:
				saved_count += 1
			print(f"  {msg}")
		except Exception as exc:
			msg = str(exc)
			failures.append((url, msg))
			print(f"  ERROR -> {msg}")
			if STOP_ON_FIRST_ERROR:
				break

	elapsed = max(time.time() - start_time, 0.001)
	rate = saved_count / elapsed
	print(
		f"\nCompleted. Saved: {saved_count}, Skipped: {skipped_count}, "
		f"Failed: {len(failures)}, Elapsed: {elapsed:.2f}s, Save rate: {rate:.2f}/s"
	)
	if failures:
		print("Failures:")
		for failed_url, reason in failures:
			print(f"  - {failed_url} :: {reason}")
		return 1

	return 0


if __name__ == "__main__":
	raise SystemExit(main())