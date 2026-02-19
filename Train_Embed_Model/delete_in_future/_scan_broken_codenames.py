"""Scan a_chunks.json for broken codenames (space inserted mid-word)."""
import json, re
from pathlib import Path

chunks_path = Path(__file__).resolve().parent.parent / "Data/framemaker/mif_jsx/a_chunks.json"
chunks = json.loads(chunks_path.read_text("utf-8"))

# Pattern: FDK prefix + partial word + space + lowercase continuation
# e.g. "FE_BadParamete r" should be "FE_BadParameter"
broken_re = re.compile(
    r"((?:Constants\.)?(?:FV|FP|FA|FO|FS|FF|FE|FTI|FT|KBD)_[A-Za-z0-9_]*[a-zA-Z])"
    r" ([a-z][a-zA-Z0-9_]*)"
)

# Also check for broken CamelCase identifiers (word + space + lowercase continuation)
# e.g. "ObjectVali d" or "TextRang e"
camel_broken_re = re.compile(
    r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)*[A-Z]?[a-z]*[a-zA-Z]) ([a-z]{1,4})\b"
)

# Broader check: any word ending with a single char then space then 1-3 lowercase chars
# that looks like a line-break artifact
linebreak_re = re.compile(r"(\b\w{4,}) ([a-z]{1,3})\b")

broken = {}
for chunk in chunks:
    content = chunk.get("content", "")
    
    for m in broken_re.finditer(content):
        key = m.group(0)
        fixed = m.group(1) + m.group(2)
        if key not in broken:
            broken[key] = {"fixed": fixed, "count": 0, "chunks": [], "type": "prefix"}
        broken[key]["count"] += 1
        if len(broken[key]["chunks"]) < 3:
            broken[key]["chunks"].append(chunk["id"])

print(f"Found {len(broken)} unique broken prefix-based patterns")
print(f"Total occurrences: {sum(v['count'] for v in broken.values())}")
print()

for b, info in sorted(broken.items(), key=lambda x: -x[1]["count"])[:50]:
    print(f"  {info['count']:3d}x  {b:50s}  ->  {info['fixed']}")

# Also look for specific known breaks
print("\n\n=== Searching for specific known broken patterns ===")
known_searches = [
    "BadParamete r", "BadOperatio n", "Surround ing", "Capitalizatio n",
    "Synchronize d", "FirstPgfInFlo w", "NextTblInDo c",
]
for pattern in known_searches:
    count = 0
    for chunk in chunks:
        if pattern in chunk.get("content", ""):
            count += 1
    if count:
        print(f"  Found '{pattern}' in {count} chunks")

# Count how many chunks have ANY occurrence of "word-space-1to3chars" at end of line
print("\n\n=== Line-break artifact analysis ===")
# Pattern: a word ending right before what was a line break, continuation on "next line"
# In the scraped content, these show up as "longwor d" or "somethin g"
artifact_re = re.compile(r"([a-zA-Z]{2,})  ?([a-z]{1,2})\b")
artifact_chunks = 0
artifact_total = 0
for chunk in chunks:
    content = chunk.get("content", "")
    matches = artifact_re.findall(content)
    if matches:
        artifact_chunks += 1
        artifact_total += len(matches)

print(f"  Chunks with potential line-break artifacts: {artifact_chunks}/{len(chunks)}")
print(f"  Total potential artifacts: {artifact_total}")
