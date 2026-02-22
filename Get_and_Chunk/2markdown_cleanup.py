import os
import sys
from pathlib import Path
from common.markdown_utils import *
# Shared metadata utilities (centralized CSV loading + merge rules)
from common.utils import *

CWD: Path = get_csv_to_process()['cwd'] # Get working directory from CSV config

# Set up global loger with script-specific CSV header; overwrite existing log
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message"]
logger = setup_global_logger(script_name=script_base, log_level='INFO', headers=LOG_HEADER)

def clean_markdown_file_inplace(md_file):
	with open(md_file, 'r', encoding='utf-8') as f:
		markdown = f.read()
	cleaned = remove_content_before_h1(markdown)
	cleaned = delete_specified_heading_content(cleaned)
	cleaned = remove_content_under_heading_below_chunk_min_threshold(cleaned)
	cleaned = fix_empty_h1(cleaned)
	cleaned = fix_no_toplevel_heading(cleaned)
	cleaned = normalize_headings(cleaned)
	cleaned = add_language_to_code_fence(cleaned)
	cleaned = remove_code_line_numbers(cleaned)
	# Apply generic regex cleanup (CSV-driven) last so earlier structural removals don't interfere
	cleaned = custom_regex(cleaned)
	# Ensure LF newlines are written to disk so CSV-driven replacements (e.g., CRLF->LF) persist.
	with open(md_file, 'w', encoding='utf-8', newline='\n') as f:
		f.write(cleaned)

if __name__ == "__main__":
	# Use CWD resolved from get_csv_to_process() at module import time
	CWD.mkdir(parents=True, exist_ok=True)
	# Find markdown files recursively (include .md, .markdown, .mdx)
	md_files = []
	for pat in ("*.md", "*.markdown", "*.mdx"):
		md_files.extend(CWD.rglob(pat))
	# Deduplicate & sort for stable processing order
	md_files = sorted(set(md_files))
	if not md_files:
		print(f"No markdown files found in {CWD}")
		sys.exit(0)
	print(f"Cleaning {len(md_files)} markdown files in {CWD} (overwriting)...")
	for md_file in md_files:
		clean_markdown_file_inplace(md_file)
		# Show path relative to the crawled output dir (parent crawled path + filename)
		try:
			display_path = md_file.relative_to(CWD)
		except Exception:
			display_path = md_file.name
		print(f"Cleaned: {display_path}")
		logger.info(f"{display_path}: cleaned")
