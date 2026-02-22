
from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, Any
from common.utils import (get_csv_to_process)

"""Shared metadata loading and front matter utilities.

Loads metadataconfig.csv once (module-level cache) and exposes helper functions
for crawlers to avoid duplication.

I AM NOT USING THIS AT THE MOMENT AS THE METADATA SIMPLY COMES FROM metadata = crawl_info['input_csv_row']
VIA get_csv_to_process.
"""

def inject_standard_metadata(chunk: dict, meta: dict) -> None:
    """
    Inject standard metadata fields from meta into the chunk dict in-place.
    Fields: title, author, category, description, tags
    """
    chunk["title"] = meta.get("METADATA_TITLE", "")
    chunk["author"] = meta.get("METADATA_AUTHOR", "")
    chunk["category"] = meta.get("METADATA_CATEGORY", "")
    chunk["description"] = meta.get("METADATA_DESCRIPTION", "")
    chunk["tags"] = meta.get("TAGS", "")

def build_front_matter_block(meta: dict) -> str:
    """
    Build a YAML front matter block from a metadata dict using FIELD_MAP_ORDER.
    Returns a string suitable for writing at the top of a markdown file or chunk.
    """
    lines = ["---"]
    for label, key in FIELD_MAP_ORDER:
        lines.append(f"{label}: {meta.get(key, '')}")
    lines.append("---\n")
    return "\n".join(lines)

# Module-level cache
_METADATA_ROWS: Dict[str, Dict[str, str]] | None = None
_CSV_PATH = 'metadataconfig.csv'

# Ordered mapping from front matter display label -> internal CSV key.
# The commented out lines appear in the input Data Table, but are not used in the front matter.
FIELD_MAP_ORDER = [
    #("Base directory", "BASE_DIR"),
    #("Source URL", "CRAWL_URL"),
    ("Title", "METADATA_TITLE"),
    ("Author", "METADATA_AUTHOR"),
    ("Category", "METADATA_CATEGORY"),
    ("Description", "METADATA_DESCRIPTION"),
    ("Tags", "TAGS"),
    #("Retrieved date", "METADATA_DATE"),
    #("Pages", "PAGES"),  # Optional; may be blank for non-PDF crawls
]

def build_front_matter(meta: Dict[str, Any], rel_path: str) -> str:
    """
    Build a YAML-style front matter block from a metadata dict.

    rel_path retained for compatibility (not currently used for a field).
    """
    lines = ["---"]
    for label, key in FIELD_MAP_ORDER:  # Preserve stable key ordering for determinism.
        lines.append(f"{label}: {meta.get(key, '')}")
    lines.append("---")
    lines.append("")  # blank separator
    return "\n".join(lines)

# -------- Per-page metadata merge utilities ---------

def _extract_leading_yaml_block(text: str) -> tuple[dict, str]:
    """If text starts with a YAML front matter block (--- ... ---),
    return (parsed_dict, remaining_text). Extremely lightweight parser that
    only handles flat key: value pairs (no nesting). If no block, returns ({}, original text).
    """
    stripped = text.lstrip()  # allow leading newlines
    if not stripped.startswith('---'):
        return {}, text
    # Find closing fence
    parts = stripped.split('\n')
    if len(parts) < 2:
        return {}, text
    block_lines = []
    # skip first fence
    for line in parts[1:]:  # Collect YAML lines until next '---'.
        if line.strip() == '---':
            break
        block_lines.append(line)
    else:  # no closing fence
        return {}, text
    # Reconstruct remainder after first closing fence
    # Find index of closing fence
    closing_index = 1 + len(block_lines)
    remainder = '\n'.join(parts[closing_index+1:])
    meta: dict = {}
    for raw in block_lines:  # Parse simple key: value pairs.
        if ':' not in raw:
            continue
        k, v = raw.split(':', 1)
        meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, remainder

def merge_page_metadata(base_meta: Dict[str, Any], raw_content: str) -> tuple[str, str]:
    """Apply user rules (updated):
    - If original has title: append to base Title with ' - <orig>' (avoid duplicate)
    - If original has description: replace base Description
    - All other original key/value pairs -> aggregated into Tags as 'key=value'
      entries separated by ", " (existing global tags preserved as first entries)
    - Remove original metadata block from body
    Returns (new_front_matter, cleaned_body)
    """
    orig_meta, body = _extract_leading_yaml_block(raw_content)  # Extract & strip original header.
    merged = dict(base_meta)  # Shallow copy to avoid mutating caller's dict.

    # --- Title merge ---
    orig_title = orig_meta.get('title') or orig_meta.get('Title')  # Support different casings.
    if orig_title:
        base_title = merged.get('METADATA_TITLE', '')
        # Only append if not already present in the standardized form
        if base_title and not base_title.endswith(f" - {orig_title}") and orig_title not in base_title.split(' - '):
            merged['METADATA_TITLE'] = f"{base_title} - {orig_title}"
        elif not base_title:
            merged['METADATA_TITLE'] = orig_title

    # --- Description replace ---
    orig_desc = orig_meta.get('description') or orig_meta.get('Description')  # Case-insensitive access.
    if orig_desc:
        merged['METADATA_DESCRIPTION'] = orig_desc

    # --- Tags aggregation as key=value ---
    tag_entries: list[str] = []  # Accumulator for tag/value strings.
    existing_tags = merged.get('TAGS', '').strip()
    if existing_tags and existing_tags not in ('[]', ''):
        # Split existing tags on common delimiters to normalize; keep raw if no delimiter
        if any(d in existing_tags for d in ['|', ',', ';']):
            parts = [p.strip() for d in ['|', ',', ';'] for p in existing_tags.split(d)]
            # The above over-splits multiple times; better do sequential split
            normalized = []
            for token in existing_tags.replace('|', ',').replace(';', ',').split(','):  # Normalize delimiters.
                token = token.strip()
                if token:
                    normalized.append(token)
            tag_entries.extend(normalized)
        else:
            tag_entries.append(existing_tags)

    for k, v in orig_meta.items():  # Convert remaining metadata into key=value pairs.
        if k.lower() in ('title', 'description'):
            continue
        if not v:
            continue
        key_clean = k.strip()
        value_clean = v.strip()
        # Build key=value; commas inside value replaced to avoid delimiter collision
        value_clean = value_clean.replace(',', ' ').strip()
        pair = f"{key_clean}={value_clean}"
        tag_entries.append(pair)

    if tag_entries:
        # Deduplicate preserving order
        seen = set()
        deduped = []  # Ordered unique list.
        for entry in tag_entries:  # Manual stable de-dup.
            if entry not in seen:
                seen.add(entry)
                deduped.append(entry)
        merged['TAGS'] = ', '.join(deduped)

    fm = build_front_matter(merged, rel_path='')
    return fm, body
