# Rationale: reduce tokens by 15%. For example, the extendscript.md file went from 338841 to 286201

from __future__ import annotations
import csv
import re
import sys
from io import StringIO
from pathlib import Path
from typing import List, Tuple

# Heading/row detection helpers
ALIGNMENT_RE = re.compile(
    r"""
    ^\s*              # optional leading whitespace
    \|?               # optional leading pipe
    \s*:?-{3,}:?\s*  # first column alignment section
    (?:\|\s*:?-{3,}:?\s*)+  # remaining columns
    \|?               # optional trailing pipe
    \s*$
    """,
    re.VERBOSE,
)
FENCE_RE = re.compile(r"^\s*```")


def _split_markdown_row(line: str) -> List[str]:
    """Split a markdown table row into cells, honoring escaped pipes."""
    trimmed = line.strip()
    if trimmed.startswith("|"):
        trimmed = trimmed[1:]
    if trimmed.endswith("|"):
        trimmed = trimmed[:-1]

    cells: List[str] = []
    current: List[str] = []
    escaped = False

    for char in trimmed:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current.clear()
            continue
        current.append(char)

    cells.append("".join(current).strip())
    return cells if cells else [""]


def _looks_like_table_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped or "|" not in stripped:
        return False
    cells = _split_markdown_row(line)
    return len(cells) >= 2 and any(cell for cell in cells)


def _collect_markdown_table(lines: List[str], start: int) -> Tuple[List[str], int]:
    table_lines = [lines[start]]
    idx = start + 1
    if idx >= len(lines):
        return table_lines, idx

    table_lines.append(lines[idx])  # alignment row
    idx += 1

    while idx < len(lines):
        candidate = lines[idx]
        if not candidate.strip():
            break
        if not _looks_like_table_row(candidate):
            break
        table_lines.append(candidate)
        idx += 1

    return table_lines, idx


def _markdown_table_to_csv_block(table_lines: List[str], *, start_line: int | None = None) -> str:
    if len(table_lines) < 2:
        return "\n".join(table_lines)

    header = _split_markdown_row(table_lines[0])
    data_rows = [_split_markdown_row(line) for line in table_lines[2:]]
    width = max((len(row) for row in ([header] + data_rows)), default=0)
    if len(header) < width:
        # Add placeholder column names so CSV stays rectangular.
        missing = width - len(header)
        loc = f"line {start_line}" if start_line else "unknown line"
        print(
            f"[mdtable2csv] Warning: table starting at {loc} has {width} columns but header lists {len(header)}."
        )
        header.extend([f"Column {i}" for i in range(len(header) + 1, len(header) + missing + 1)])

    rows = [header] + data_rows

    for row in rows:
        if len(row) < width:
            row.extend([""] * (width - len(row)))
        elif len(row) > width:
            # Should not happen, but guard anyway
            row[:] = row[:width]

    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        writer.writerow(row)

    csv_text = buffer.getvalue().strip()
    return f"<!-- Markdown Table -->\n{csv_text}\n"


def convert_markdown_tables_to_csv(md_path: Path) -> Path:
    """Replace markdown pipe tables with CSV blocks and write a sibling file."""
    source = md_path.read_text(encoding="utf-8")
    lines = source.splitlines()

    idx = 0
    in_fence = False
    output_lines: List[str] = []

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        if FENCE_RE.match(stripped):
            in_fence = not in_fence
            output_lines.append(line)
            idx += 1
            continue

        if (
            not in_fence
            and idx + 1 < len(lines)
            and _looks_like_table_row(line)
            and ALIGNMENT_RE.match(lines[idx + 1].strip())
        ):
            table_start_line = idx + 1
            table_lines, idx = _collect_markdown_table(lines, idx)
            csv_block = _markdown_table_to_csv_block(table_lines, start_line=table_start_line)
            output_lines.extend(csv_block.rstrip("\n").split("\n"))
            continue

        output_lines.append(line)
        idx += 1

    result = "\n".join(output_lines)
    if source.endswith("\n"):
        result += "\n"

    out_path = md_path.with_name(f"{md_path.stem}_mdtables_as_csv.md")
    out_path.write_text(result, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mdtable2csv.py <markdown_file>")
        sys.exit(1)

    md_file = Path(sys.argv[1])
    if not md_file.exists():
        print(f"Error: file not found: {md_file}")
        sys.exit(1)

    output = convert_markdown_tables_to_csv(md_file)
    print(f"✅ Wrote: {output}")
