"""
3.01add_codefriendly_names2chunk.py — Apply codename-to-friendly-name mappings to chunks.

Reads the reviewed CSV produced by 3.01codename_to_friendlyname.py and a_chunks.json,
scans each chunk's "content" field for matching FDK codenames, and adds a
"code_friendly_name" field containing semicolon-delimited "CODENAME: friendly name"
pairs. Chunks with no matches get an empty string. No existing fields are modified.

Prerequisites:
    - codename_friendlyname.csv must exist (run 3.01codename_to_friendlyname.py first)
      Falls back to 3.02friendlynameoutput.csv if the primary CSV is not found.
    - a_chunks.json must exist in the CWD (run 3.00chunker.py first)

Pipeline conventions:
    - Uses get_csv_to_process() for CWD resolution from metadataconfig.csv
    - Uses setup_global_logger() for CSV-formatted logging
    - Creates a .json.bak backup before mutating a_chunks.json

Implementation steps:
    1. Bootstrap: get_csv_to_process() -> CWD, setup logger
    2. Load codename_friendlyname.csv into a codename->friendlyname dict
       (skip rows with blank friendlyname)
    3. Build search variants for each codename:
       - Add Constants.{codename} for FV_/FP_/FA_/FO_/FS_/FF_/FE_/FTI_/FT_ prefixes
       - Add FCodes.{codename} for KBD_ prefixes
    4. Compile a single regex from all search forms (longest-first, re.escaped)
    5. Load a_chunks.json
    6. For each chunk, regex-search "content", collect matched codenames,
       set chunk["code_friendly_name"] = sorted semicolon-joined "CN: friendly" pairs
    7. Backup a_chunks.json -> .json.bak, then write updated JSON

Usage:
    python 3.01add_codefriendly_names2chunk.py
"""

import csv
import json
import os
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

from common.utils import setup_global_logger

# ── Bootstrap ─────────────────────────────────────────────────────────────────
CWD = Path(r"C:\GIT\Z_Master_Rag\Data\framemaker\mif_jsx")

CHUNKS_PATH = CWD / "a_chunks.json"

# CSV produced by 3.01codename_to_friendlyname.py (primary)
# Falls back to 3.02friendlynameoutput.csv (alternate) if primary not found
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PRIMARY = SCRIPT_DIR / "codename_friendlyname.csv"
CSV_FALLBACK = SCRIPT_DIR / "3.02friendlynameoutput.csv"

# Import the plain-English expansion function from the companion script.
# Can't use importlib directly because the companion has a bare TODO syntax error.
# Instead, extract and exec the relevant dicts + functions.
_COMPANION = SCRIPT_DIR / "3.01codename_to_friendlyname.py"
_companion_src = _COMPANION.read_text(encoding="utf-8")

# Extract PLAIN_ENGLISH_OVERRIDES, WORD_EXPANSIONS, COMPOUND_SPLITS,
# _expand_word(), and _auto_expand_phrase() by line range.
_lines = _companion_src.splitlines(keepends=True)
# Find boundaries
_pe_start = next(i for i, l in enumerate(_lines) if "PLAIN_ENGLISH_OVERRIDES" in l and "=" in l)
_we_start = next(i for i, l in enumerate(_lines) if "WORD_EXPANSIONS" in l and "=" in l)
_cs_start = next(i for i, l in enumerate(_lines) if "COMPOUND_SPLITS" in l and "=" in l)
_fn_start = next(i for i, l in enumerate(_lines) if "def _expand_word" in l)
# _auto_expand_phrase ends at CONVERT_OUTPUT_PATH line
_fn_end = next(i for i, l in enumerate(_lines) if "CONVERT_OUTPUT_PATH" in l)

_exec_block = "".join(
    _lines[_pe_start:_fn_end]
)
_ns: dict = {}
exec(_exec_block, _ns)
expand_to_plain_english = _ns["_auto_expand_phrase"]

# Minimum character length for a friendly name to be kept (filters noise like "tr", "ca")
MIN_NAME_LENGTH = 4

# Logger
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "Mappings Loaded", "Chunks Enriched"]
logger = setup_global_logger(script_name=script_base, log_level='INFO', headers=LOG_HEADER)


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_csv_path() -> Path:
    """Return the CSV path to use, preferring the primary over the fallback."""
    if CSV_PRIMARY.exists():
        return CSV_PRIMARY
    if CSV_FALLBACK.exists():
        logger.info(f"Primary CSV not found, using fallback: {CSV_FALLBACK.name}")
        return CSV_FALLBACK
    logger.error(f"No CSV found. Looked for:\n  {CSV_PRIMARY}\n  {CSV_FALLBACK}")
    sys.exit(1)


def load_mappings(csv_path: Path) -> dict[str, str]:
    """Load codename -> friendlyname mappings from CSV.

    Each friendly name is expanded to plain English via the companion script's
    expansion logic (PLAIN_ENGLISH_OVERRIDES + WORD_EXPANSIONS).  Entries whose
    expanded name is shorter than MIN_NAME_LENGTH are dropped as noise.
    """
    mappings: dict[str, str] = {}
    skipped = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cn = row["codename"].strip()
            fn = row["friendlyname"].strip()
            if not cn or not fn:
                continue
            expanded = expand_to_plain_english(fn)
            if len(expanded) < MIN_NAME_LENGTH:
                skipped += 1
                continue
            mappings[cn] = expanded
    logger.info(f"Loaded {len(mappings)} mappings from {csv_path.name} "
                f"(skipped {skipped} short names < {MIN_NAME_LENGTH} chars)")
    return mappings


def build_regex(mappings: dict[str, str]) -> tuple[re.Pattern, dict[str, str]]:
    """Build a compiled regex and a search-form -> canonical-codename lookup.

    For each codename, adds search variants:
      - Constants.{codename}  for FV_/FP_/FA_/FO_/FS_/FF_/FE_/FTI_/FT_ prefixes
      - FCodes.{codename}     for KBD_ prefixes
    Sorts longest-first to prevent partial matches.
    """
    search_to_canonical: dict[str, str] = {}
    for cn in mappings:
        search_to_canonical[cn] = cn
        if cn.startswith(("FV_", "FP_", "FA_", "FO_", "FS_",
                          "FF_", "FE_", "FTI_", "FT_")):
            search_to_canonical[f"Constants.{cn}"] = cn
        if cn.startswith("KBD_"):
            search_to_canonical[f"FCodes.{cn}"] = cn

    sorted_forms = sorted(search_to_canonical, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(f) for f in sorted_forms))
    return pattern, search_to_canonical


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load chunks from JSON file."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_path.name}")
    return chunks


def apply_friendly_names(
    chunks: list[dict],
    mappings: dict[str, str],
    pattern: re.Pattern,
    search_to_canonical: dict[str, str],
) -> list[dict]:
    """Add 'code_friendly_name' field to each chunk.

    Scans the chunk's 'content' for codenames matching the compiled regex.
    Sets code_friendly_name to sorted semicolon-joined friendly names only,
    or '' if no matches.
    """
    enriched_count = 0
    result = []
    for chunk in chunks:
        content = chunk.get("content", "")
        found_canonical: set[str] = set()
        for m in pattern.finditer(content):
            found_canonical.add(search_to_canonical[m.group()])

        if found_canonical:
            names = sorted(set(mappings[cn] for cn in found_canonical))
            value = "; ".join(names)
            enriched_count += 1
        else:
            value = ""

        # Insert code_friendly_name directly after "content"
        new_chunk = OrderedDict()
        for key, val in chunk.items():
            if key == "code_friendly_name":
                continue  # skip old value; we insert fresh below
            new_chunk[key] = val
            if key == "content":
                new_chunk["code_friendly_name"] = value
        # Safety: if "content" key was missing, append at end
        if "code_friendly_name" not in new_chunk:
            new_chunk["code_friendly_name"] = value
        result.append(dict(new_chunk))

    logger.info(f"Enriched {enriched_count}/{len(chunks)} chunks with code_friendly_name")
    return result


def save_chunks(chunks: list[dict], chunks_path: Path) -> None:
    """Backup the original JSON, then write the updated chunks."""
    backup_path = chunks_path.with_suffix(".json.bak")
    shutil.copy2(chunks_path, backup_path)
    logger.info(f"Backup saved to {backup_path.name}")

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved enriched chunks to {chunks_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("Starting code_friendly_name enrichment")

    # 1. Resolve CSV
    csv_path = resolve_csv_path()

    # 2. Load mappings
    mappings = load_mappings(csv_path)
    if not mappings:
        logger.error("No mappings loaded — nothing to apply. Exiting.")
        sys.exit(1)

    # 3. Build regex
    pattern, search_to_canonical = build_regex(mappings)

    # 4. Load chunks
    if not CHUNKS_PATH.exists():
        logger.error(f"{CHUNKS_PATH} not found. Run 3.00chunker.py first.")
        sys.exit(1)
    chunks = load_chunks(CHUNKS_PATH)

    # 5. Apply friendly names
    chunks = apply_friendly_names(chunks, mappings, pattern, search_to_canonical)

    # 6. Save
    save_chunks(chunks, CHUNKS_PATH)

    logger.info("Done.")


if __name__ == "__main__":
    main()
