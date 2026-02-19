"""_fix_broken_codenames.py  --  Fix line-break artifacts in a_chunks.json

Scans chunk content for FDK constants that were split by a space due to
line-wrapping during PDF/web crawling (e.g. "FE_BadParamete r" -> "FE_BadParameter").

Strategy:
  1. Collect all INTACT FDK constants in the corpus (the "known good" set)
  2. For each candidate break (prefix + partial + space + continuation),
     check if joining creates a token that exists in the known-good set
  3. Also use heuristics: if the right-side fragment is 1-3 chars and NOT
     a common English word, it's almost certainly a break
  4. Write a report, then optionally apply fixes

Usage:
    python _fix_broken_codenames.py              # Scan and report
    python _fix_broken_codenames.py --apply      # Apply fixes to a_chunks.json

NOTE for future use: this has a bug. I had to run (F[A-Za-z]+_[A-Za-z_]+) ([A-Z])
after and manually fix 200+ strings.
"""

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

CHUNKS_PATH = (
    Path(__file__).resolve().parent.parent
    / "Data" / "framemaker" / "mif_jsx" / "a_chunks.json"
)

# Common short English words that appear after codenames in normal prose
# These should NOT be "joined" back into the codename
ENGLISH_WORDS = {
    # 1-char
    "a", "i",
    # 2-char
    "an", "as", "at", "be", "by", "do", "go", "if", "in", "is", "it",
    "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we",
    # 3-char
    "all", "and", "any", "are", "but", "can", "did", "end", "for", "get",
    "got", "had", "has", "her", "him", "his", "how", "its", "let", "may",
    "new", "nor", "not", "now", "off", "old", "one", "our", "out", "own",
    "per", "put", "ran", "run", "say", "set", "she", "the", "too", "two",
    "use", "was", "way", "who", "why", "win", "won", "yes", "yet", "you",
    # 4-char common after codenames
    "also", "both", "does", "each", "else", "from", "have", "here",
    "into", "just", "like", "more", "most", "must", "name", "none",
    "once", "only", "over", "same", "some", "such", "than", "that",
    "them", "then", "they", "this", "thus", "type", "upon", "used",
    "very", "were", "what", "when", "will", "with", "your",
    # 5+ char common words that follow codenames in prose
    "after", "allow", "below", "being", "cause", "class",
    "comma", "could", "marks", "might", "never", "other",
    "point", "right", "shall", "since", "still", "their",
    "there", "these", "thing", "those", "under", "until",
    "using", "where", "which", "while", "whose", "would",
    "about", "above", "array", "based", "check", "class",
    "comma", "could", "every", "field", "first", "given",
    "holds", "means", "needs", "never", "other", "panel",
    "range", "reads", "refer", "returns", "should", "shows",
    "since", "start", "store", "table", "takes", "their",
    "thing", "those", "token", "valid", "which", "works",
    "write", "define", "during", "enable", "manage",
    "return", "string", "command", "control", "returns",
    "contain", "default", "display", "element", "include",
    "integer", "objects", "options", "removes", "setting",
    "specify", "support", "without", "constants",
    # FDK-specific words that follow constants in prose
    "flag", "value", "object", "property", "method", "function",
    "error", "code", "index", "data", "list", "item", "text",
    "node", "true", "false", "null", "void", "etc",
}

# ── Regex patterns for intact codenames (no spaces) ───────────────────────────
RE_INTACT_PREFIX = re.compile(
    r"(?:Constants\.)?(?:FV|FP|FA|FO|FS|FF|FE|FTI|FT)_[A-Za-z0-9_]+"
)
RE_INTACT_FAPI = re.compile(r"\bF_Api[A-Za-z]+\b")
RE_INTACT_STRUCT = re.compile(r"\bF_[A-Z][a-zA-Z]+T\b")
RE_INTACT_FCODES = re.compile(r"FCodes\.[A-Za-z0-9_]+")

# ── Regex for candidate breaks ────────────────────────────────────────────────
# Prefix constant + partial word + space + lowercase continuation
RE_BROKEN_PREFIX = re.compile(
    r"((?:Constants\.)?(?:FV|FP|FA|FO|FS|FF|FE|FTI|FT)_[A-Za-z0-9_]*[a-zA-Z])"
    r" ([a-z][a-zA-Z0-9]*)"
)
# F_Api function with break
RE_BROKEN_FAPI = re.compile(
    r"(F_Api[A-Za-z]+) ([a-z][a-zA-Z0-9]*)"
)
# FCodes with break
RE_BROKEN_FCODES = re.compile(
    r"(FCodes\.[A-Za-z0-9_]+) ([a-z][a-zA-Z0-9]*)"
)
# General CamelCase word break (PascalCase word + space + 1-3 lowercase chars)
RE_BROKEN_CAMEL = re.compile(
    r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+[A-Z]?[a-z]*[a-zA-Z]) ([a-z]{1,3})\b"
)


def load_chunks() -> list[dict]:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_intact_codenames(chunks: list[dict]) -> set[str]:
    """Collect all intact (non-broken) FDK constants across the corpus."""
    intact = set()
    for chunk in chunks:
        content = chunk.get("content", "")
        for pattern in (RE_INTACT_PREFIX, RE_INTACT_FAPI,
                        RE_INTACT_STRUCT, RE_INTACT_FCODES):
            for m in pattern.finditer(content):
                intact.add(m.group())
    return intact


def find_breaks(chunks: list[dict], intact: set[str]) -> list[dict]:
    """Find all line-break artifacts in the corpus.

    Returns list of dicts: {broken, fixed, chunk_id, start, end, confidence, reason}
    """
    fixes = []

    for chunk in chunks:
        content = chunk.get("content", "")
        chunk_id = chunk["id"]

        for pattern in (RE_BROKEN_PREFIX, RE_BROKEN_FAPI, RE_BROKEN_FCODES):
            for m in pattern.finditer(content):
                left = m.group(1)
                right = m.group(2)
                broken = m.group(0)
                joined = left + right

                # Skip if right side is a common English word
                if right.lower() in ENGLISH_WORDS:
                    continue

                # Determine confidence
                # Strip Constants. prefix for intact lookup
                joined_bare = joined.replace("Constants.", "")
                left_bare = left.replace("Constants.", "")

                if joined in intact or joined_bare in intact:
                    confidence = "high"
                    reason = "joined form exists intact elsewhere"
                elif len(right) <= 3:
                    confidence = "high"
                    reason = f"short fragment ({len(right)} chars), not a word"
                elif joined.replace("Constants.", "").replace("FCodes.", "") in intact:
                    confidence = "high"
                    reason = "joined form exists (stripped accessor)"
                elif left.startswith("Constants.") or left.startswith("FCodes."):
                    # If it has an explicit accessor prefix, the left side
                    # is clearly an FDK constant -- any non-English continuation
                    # is a line-break artifact
                    confidence = "high"
                    reason = "accessor-prefixed constant + non-English continuation"
                else:
                    confidence = "medium"
                    reason = "heuristic: partial word + continuation"

                fixes.append({
                    "broken": broken,
                    "fixed": joined,
                    "chunk_id": chunk_id,
                    "start": m.start(),
                    "end": m.end(),
                    "confidence": confidence,
                    "reason": reason,
                })

    return fixes


def apply_fixes(chunks: list[dict], fixes: list[dict]) -> tuple[list[dict], int]:
    """Apply text replacements to chunk content."""
    # Group fixes by chunk_id
    by_chunk: dict[str, list[dict]] = {}
    for f in fixes:
        by_chunk.setdefault(f["chunk_id"], []).append(f)

    total_applied = 0
    for chunk in chunks:
        chunk_fixes = by_chunk.get(chunk["id"], [])
        if not chunk_fixes:
            continue

        content = chunk["content"]
        # Sort by position descending to avoid offset shifts
        chunk_fixes.sort(key=lambda f: f["start"], reverse=True)

        for fix in chunk_fixes:
            # Verify the broken text is still at the expected position
            actual = content[fix["start"]:fix["end"]]
            if actual == fix["broken"]:
                content = content[:fix["start"]] + fix["fixed"] + content[fix["end"]:]
                total_applied += 1
            else:
                # Position shifted, do string replace (first occurrence)
                if fix["broken"] in content:
                    content = content.replace(fix["broken"], fix["fixed"], 1)
                    total_applied += 1

        chunk["content"] = content

    return chunks, total_applied


def main():
    parser = argparse.ArgumentParser(description="Fix broken FDK codenames")
    parser.add_argument("--apply", action="store_true",
                        help="Apply fixes to a_chunks.json")
    args = parser.parse_args()

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    intact = collect_intact_codenames(chunks)
    print(f"Found {len(intact)} intact codenames in corpus")

    fixes = find_breaks(chunks, intact)
    print(f"Found {len(fixes)} candidate fixes")

    # Deduplicate by (broken, fixed) for reporting
    unique_fixes = {}
    for f in fixes:
        key = (f["broken"], f["fixed"])
        if key not in unique_fixes:
            unique_fixes[key] = {
                "count": 0,
                "confidence": f["confidence"],
                "reason": f["reason"],
            }
        unique_fixes[key]["count"] += 1

    # Report
    high = [(k, v) for k, v in unique_fixes.items() if v["confidence"] == "high"]
    med = [(k, v) for k, v in unique_fixes.items() if v["confidence"] == "medium"]

    print(f"\nHigh confidence: {len(high)} unique ({sum(v['count'] for _, v in high)} occurrences)")
    print(f"Medium confidence: {len(med)} unique ({sum(v['count'] for _, v in med)} occurrences)")

    print(f"\n{'='*80}")
    print("HIGH CONFIDENCE FIXES")
    print(f"{'='*80}")
    for (broken, fixed), info in sorted(high, key=lambda x: -x[1]["count"])[:60]:
        print(f"  {info['count']:3d}x  {broken:55s} -> {fixed}")

    if med:
        print(f"\n{'='*80}")
        print("MEDIUM CONFIDENCE FIXES (review these)")
        print(f"{'='*80}")
        for (broken, fixed), info in sorted(med, key=lambda x: -x[1]["count"])[:40]:
            print(f"  {info['count']:3d}x  {broken:55s} -> {fixed}")

    if args.apply:
        # Only apply high-confidence fixes
        high_fixes = [f for f in fixes if f["confidence"] == "high"]
        print(f"\n\nApplying {len(high_fixes)} high-confidence fixes...")

        backup = CHUNKS_PATH.with_suffix(".json.pre_fix_bak")
        shutil.copy2(CHUNKS_PATH, backup)
        print(f"Backup saved to {backup}")

        chunks, applied = apply_fixes(chunks, high_fixes)
        print(f"Applied {applied}/{len(high_fixes)} fixes")

        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved to {CHUNKS_PATH}")

        # Verify: re-scan
        chunks2 = load_chunks()
        fixes2 = find_breaks(chunks2, collect_intact_codenames(chunks2))
        remaining_high = sum(1 for f in fixes2 if f["confidence"] == "high")
        print(f"\nPost-fix scan: {len(fixes2)} candidates remain ({remaining_high} high confidence)")
    else:
        print(f"\nRun with --apply to fix high-confidence breaks")


if __name__ == "__main__":
    main()
