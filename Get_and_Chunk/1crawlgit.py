"""GitHub targeted Markdown fetcher.

Purpose:
    Download ONLY the specified directory (and its descendants) of a GitHub
    repository (Markdown files *.md) and write them under MARKDOWN_STORE/BASE_DIR
    preserving relative paths beneath that directory.

High‑level flow:
    1. Load crawl metadata row via CRAWL_ID (provides CRAWL_URL + global fields).
    2. Parse the GitHub URL -> (owner, repo, branch, subpath).
    3. List all markdown files beneath that subpath using the GitHub Contents API.
    4. For each file, download raw content, merge any original YAML front matter
         into global metadata (title append, description replace, residual -> Tags).
    5. Emit a normalized front matter block + cleaned body.

Note: Some legacy duplicate helpers remain until full de‑dup cleanup step.
"""
import requests
from pathlib import Path
import os
from urllib.parse import urlparse
# Shared metadata utilities (centralized CSV loading + merge rules)
from common.metadata_utils import merge_page_metadata
from common.utils import (get_csv_to_process, setup_global_logger)

csvrow_data = get_csv_to_process() # Get the entire csv row to process, based dir, url, user metadata, etc. 
metadata = csvrow_data['input_csv_row'] # Store the row data in a var
CWD: Path = csvrow_data['cwd'] # Extract the rootdir/basedir from the csv row data
CRAWL_URL = metadata['CRAWL_URL'] # Get the URL rom the CRAWL_URL field in the csv row

# Set up global loger with script-specific CSV header; overwrite existing log
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "Filename", "Token Count"]
logger = setup_global_logger(script_name=script_base, log_level='INFO', headers=LOG_HEADER)

def parse_github_subdir_url(crawl_url: str):
    """Parse a GitHub repository (optionally with /tree/<branch>/<subpath>) URL.

    Supports forms:
      - https://github.com/owner/repo
      - https://github.com/owner/repo/
      - https://github.com/owner/repo/tree/<branch>
      - https://github.com/owner/repo/tree/<branch>/<subpath>

    Returns:
        (owner, repo, branch, subpath)
    Defaults:
        branch = 'main' if not specified; subpath = '' (root of repository).
    Raises:
        ValueError if URL path cannot yield owner & repo.
    """
    parsed = urlparse(crawl_url)
    parts = [p for p in parsed.path.split('/') if p]
    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub repo URL: {crawl_url}")
    owner = parts[0]
    repo = parts[1].removesuffix('.git')
    branch = 'main'
    subpath = ''
    if len(parts) >= 4 and parts[2] == 'tree':
        branch = parts[3] or 'main'
        if len(parts) > 4:
            subpath = '/'.join(parts[4:])
    return owner, repo, branch, subpath


def list_markdown_files(owner: str, repo: str, subpath: str, branch: str = 'main'):
    """Return metadata dicts for all markdown (*.md) files under a given subpath.

    Strategy: Iterative DFS/BFS hybrid using a stack (to_visit). Each GitHub
    directory listing returns JSON with 'file' or 'dir' entries. We queue dirs
    and collect files that end with .md (case-insensitive).
    """
    api_root = f"https://api.github.com/repos/{owner}/{repo}/contents"
    to_visit = [subpath.strip('/')] if subpath else ['']
    files = []
    seen_dirs = set()
    headers = {'Accept': 'application/vnd.github.v3+json'}
    token = os.getenv('GITHUB_PAT') or os.getenv('GITHUB_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'

    while to_visit:  # Traverse pending directories (LIFO -> depth-first style ordering).
        current = to_visit.pop()
        if current in seen_dirs:
            continue
        seen_dirs.add(current)
        url = f"{api_root}/{current}" if current else api_root
        url += f"?ref={branch}"
        resp = requests.get(url, headers=headers, timeout=(5, 60))
        if resp.status_code == 404:
            print(f"Warning: path not found: {current}")
            continue
        resp.raise_for_status()
        data = resp.json()
        # If a single file is requested (unlikely for directory crawl)
        if isinstance(data, dict) and data.get('type') == 'file':
            if data['name'].lower().endswith('.md'):
                files.append(data)
            continue
        # Otherwise list of entries
        for entry in data:  # Iterate each item in the directory listing.
            etype = entry.get('type')
            if etype == 'dir':
                to_visit.append(entry['path'])
            elif etype == 'file' and entry['name'].lower().endswith('.md'):
                files.append(entry)
    return files

def save_markdown_files(files_meta, owner: str, repo: str, subpath: str, out_root: Path, meta: dict, branch: str = 'main'):
    """Download each markdown file and write it with merged front matter.

    Args:
        files_meta: List of GitHub API file metadata dicts (must include 'path' and optionally 'download_url').
        owner/repo: Repo coordinates.
        subpath: Root subdirectory we originally enumerated (used for trimming relative paths).
        out_root: Destination base path on local filesystem.
        meta: Global crawl metadata row (CSV transformed dict).
        branch: Branch name for raw content retrieval when download_url absent.
    Returns:
        List[Path] of saved file paths.

    Notes:
        - We re-construct raw URL if GitHub API response lacks 'download_url'.
        - merge_page_metadata handles YAML extraction + merging logic.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    saved = []
    base_prefix = subpath.strip('/') + '/' if subpath else ''
    for item in files_meta:  # Process each discovered markdown file.
        raw_url = item.get('download_url')
        if not raw_url:
            # Build raw URL manually using the correct branch
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{item['path']}"
        rel_inside = item['path']
        if base_prefix and rel_inside.startswith(base_prefix):
            rel_inside = rel_inside[len(base_prefix):]
        dest = out_root / rel_inside
        dest.parent.mkdir(parents=True, exist_ok=True)

        r = requests.get(raw_url, timeout=(5, 60))
        r.raise_for_status()
        raw_content = r.text
        # Merge original page-level front matter into global metadata per rules
        # Use a per-file Source URL (HTML blob path) for better traceability
        file_meta = dict(meta)
        file_meta['CRAWL_URL'] = f"https://github.com/{owner}/{repo}/blob/{branch}/{item['path']}"
        merged_front, cleaned_body = merge_page_metadata(file_meta, raw_content)
        dest.write_text(merged_front + cleaned_body, encoding='utf-8')
        saved.append(dest)
        # Compute a simple token count (word-split). Replace with a tokenizer
        # if you need exact parity with embedding/token usage downstream.
        try:
            token_count = max(0, len((cleaned_body or "").split()))
        except Exception:
            token_count = 0

        # Log a CSV row for this saved file. The CSVFormatter in common.utils
        # will map the 'Filename' and 'Token Count' extras to the declared
        # header columns in LOG_HEADER.
        try:
            rel_path = str(dest.relative_to(CWD)).replace('\\', '/')
        except Exception:
            rel_path = str(dest)
        # Keep the Message column short but include repo and branch for context.
        # Example Message: "saved github:owner/repo@branch"
        msg = f"saved github:{owner}/{repo}@{branch}"
        try:
            logger.info(msg, extra={"Filename": rel_path, "Token Count": token_count})
        except Exception:
            # Fallback: include filename in the plain log line since extras
            # may not be supported in the fallback handler.
            logger.info(f"saved {rel_path} tokens={token_count}")
    return saved

def main():
    """Entry point for standalone execution.

    Orchestrates: metadata row load -> URL parse -> enumerate files -> download & save.
    Emits summary counts to stdout for simple monitoring/log piping.
    """
    owner, repo, branch, subpath = parse_github_subdir_url(CRAWL_URL)
    logger.info(f"Fetching markdown from {owner}/{repo}:{branch} subpath='{subpath or '.'}' -> {CWD}")
    files_meta = list_markdown_files(owner, repo, subpath, branch=branch)
    logger.info(f"Discovered {len(files_meta)} markdown files")
    if not files_meta and branch != 'main':
        logger.warning("No files found; if you intended main branch ensure URL uses /tree/main/ or path exists in chosen branch.")
    saved = save_markdown_files(files_meta, owner, repo, subpath, CWD, metadata, branch=branch)
    logger.info(f"Saved {len(saved)} to {CWD}")

if __name__ == '__main__':
    main()

