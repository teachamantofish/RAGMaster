"""Interactively move "Error returned to FA_errno" tables to the end of their headings.

Usage:
    python AI_RAG/scripts/reposition_error_tables.py [--path path/to/file.md]

Workflow (per table):
1. Press Enter to locate the next error table (prints heading + table snippet).
2. Press Enter again to cut the table and paste it at the end of the current heading.
   - The script writes the updated file immediately so you can inspect it in your editor.
3. Repeat: the next Enter finds the following table. Type "s" to skip the current table
   or "q" to quit at any prompt.

Tables are detected by the canonical header row:
    Error returned to FA_errno,Reason
and the script assumes the table begins with a "<!-- Data Table -->" marker.
"""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import List, Optional, Tuple

ERROR_HEADER = "error returned to fa_errno"
TABLE_MARKER = "<!-- data table"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)")


class TableHit:
    """Representation of a detected error table within a heading."""

    def __init__(
        self,
        heading_idx: int,
        heading_text: str,
        heading_level: int,
        table_start: int,
        table_end: int,
    ) -> None:
        self.heading_idx = heading_idx
        self.heading_text = heading_text
        self.heading_level = heading_level
        self.table_start = table_start
        self.table_end = table_end

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"TableHit(heading_idx={self.heading_idx}, heading='{self.heading_text}', "
            f"level={self.heading_level}, start={self.table_start}, end={self.table_end})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively move FA_errno error tables")
    parser.add_argument(
        "--path",
        default="Data/framemaker/extendscript/extendscript.md",
        help="Path to the markdown file to edit (default: extendscript guide)",
    )
    return parser.parse_args()


def read_lines(path: pathlib.Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    # Preserve existing newlines so formatting stays untouched.
    return text.splitlines(keepends=True)


def write_lines(path: pathlib.Path, lines: List[str]) -> None:
    path.write_text("".join(lines), encoding="utf-8")


def find_heading_above(lines: List[str], start_idx: int) -> Optional[Tuple[int, str, int]]:
    for idx in range(start_idx, -1, -1):
        match = HEADING_RE.match(lines[idx])
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            return idx, text, level
    return None


def find_heading_end(lines: List[str], heading_idx: int, heading_level: int) -> int:
    idx = heading_idx + 1
    while idx < len(lines):
        match = HEADING_RE.match(lines[idx])
        if match and len(match.group(1)) <= heading_level:
            break
        idx += 1
    return idx


def _normalize_header(line: str) -> str:
    normalized = line.strip().lower()
    normalized = normalized.replace(r"\_", "_")
    # Collapse multiple spaces to a single space so wrapped headers still match.
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def detect_error_table(lines: List[str], start_idx: int) -> Optional[TableHit]:
    idx = start_idx
    while idx < len(lines):
        marker = lines[idx].strip().lower()
        if marker.startswith(TABLE_MARKER):
            cursor = idx + 1
            while cursor < len(lines) and not lines[cursor].strip():
                cursor += 1
            if cursor < len(lines) and _normalize_header(lines[cursor]).startswith(ERROR_HEADER):
                heading = find_heading_above(lines, idx)
                if heading is None:
                    idx += 1
                    continue
                heading_idx, heading_text, heading_level = heading
                end_cursor = cursor + 1
                while end_cursor < len(lines) and lines[end_cursor].strip():
                    end_cursor += 1
                # Keep one trailing blank line (if absent, add one later when moving).
                if end_cursor < len(lines) and not lines[end_cursor].strip():
                    end_cursor += 1
                return TableHit(
                    heading_idx=heading_idx,
                    heading_text=heading_text,
                    heading_level=heading_level,
                    table_start=idx,
                    table_end=end_cursor,
                )
        idx += 1
    return None


def ensure_trailing_blank(chunk: List[str]) -> None:
    if not chunk:
        return
    if chunk[-1].strip():
        chunk.append("\n")
    # Always add one extra blank line so the table is visually separated
    chunk.append("\n")


def move_table_to_heading_end(lines: List[str], hit: TableHit) -> Tuple[int, int]:
    chunk = lines[hit.table_start:hit.table_end]
    ensure_trailing_blank(chunk)
    del lines[hit.table_start:hit.table_end]

    heading_end = find_heading_end(lines, hit.heading_idx, hit.heading_level)
    # Insert a blank line before the table if the preceding line has content and
    # is not the heading itself.
    if heading_end > 0 and lines[heading_end - 1].strip():
        chunk.insert(0, "\n")
    insert_idx = heading_end
    lines[insert_idx:insert_idx] = chunk
    new_end = insert_idx + len(chunk)
    return insert_idx, new_end


def print_table(lines: List[str], hit: TableHit) -> None:
    heading_line = hit.heading_idx + 1
    start_line = hit.table_start + 1
    end_line = hit.table_end
    print("\n=== Error Table Preview ===")
    print(f"Heading (line {heading_line}): {hit.heading_text}")
    print(f"Table lines {start_line}-{end_line}")
    print("---------------------------")
    for line in lines[hit.table_start:hit.table_end]:
        print(line.rstrip("\n"))
    print("===========================\n")


def interactive_loop(path: pathlib.Path) -> None:
    lines = read_lines(path)
    search_idx = 0
    pending: Optional[TableHit] = None
    print(f"Loaded {path}. Press Enter to locate the first error table, 'q' to quit.")

    while True:
        if pending is None:
            user = input("[Enter=find next | q=quit]: ").strip().lower()
            if user == "q":
                break
            hit = detect_error_table(lines, search_idx)
            if not hit:
                print("No more error tables found. You're done!")
                break
            print_table(lines, hit)
            pending = hit
            search_idx = hit.table_end
        else:
            user = input("[Enter=move table | s=skip | q=quit]: ").strip().lower()
            if user == "q":
                break
            if user == "s":
                print("Skipped. Press Enter to find the next table.")
                pending = None
                continue
            insert_idx, new_end = move_table_to_heading_end(lines, pending)
            write_lines(path, lines)
            print(
                f"Moved table under '{pending.heading_text}' to lines {insert_idx + 1}-{new_end}."
            )
            print("Review the update in your editor, then press Enter to continue.")
            pending = None
            search_idx = new_end

    print("Exiting. Remember to run git diff to review all changes.")


def main() -> None:
    args = parse_args()
    target = pathlib.Path(args.path).resolve()
    if not target.exists():
        raise SystemExit(f"File not found: {target}")
    interactive_loop(target)


if __name__ == "__main__":
    main()
