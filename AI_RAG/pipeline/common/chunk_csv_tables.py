"""Split CSV-style markdown tables into <=490-token chunks.

The script scans for `<!-- Data Table -->` markers, counts tokens for the
following CSV table (header + rows), and splits oversized tables into
multiple chunks while repeating the marker and header for each chunk.
Usage:
    python chunk_csv_tables.py path/to/input.md [--max-tokens 490] [--output out.md]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")

# Table-detection and split settings
TABLE_MARKER = "<!-- Data Table -->"
DEFAULT_MAX_TOKENS = 490
BALANCED_SPLIT_THRESHOLD = 900  # tables below this size try an even split
BALANCED_SPLIT_RATIO = 0.5      # fallback chunk ratio when balanced split fails


def count_tokens(text: str) -> int:
    """Token count helper using GPT-4/GPT-3.5 compatible encoding."""
    if not text:
        return 0
    return len(ENCODING.encode(text))


@dataclass
class ChunkStat:
    rows: int
    tokens: int
    overflow: bool = False


@dataclass
class TableStat:
    index: int
    start_line: int
    original_tokens: int
    original_rows: int
    chunks: List[ChunkStat]


def _chunk_token_count(header: str, rows: Sequence[str]) -> int:
    lines = [TABLE_MARKER, header]
    lines.extend(rows)
    return count_tokens("\n".join(lines))


def _build_chunk_lines(header: str, rows: Sequence[str]) -> List[str]:
    chunk_lines = [TABLE_MARKER, header]
    chunk_lines.extend(rows)
    return chunk_lines


def split_table_balanced(
    header: str,
    rows: Sequence[str],
    token_limit: int,
) -> Optional[Tuple[List[List[str]], List[ChunkStat]]]:
    """Attempt to produce two contiguous chunks with similar token sizes."""
    if len(rows) < 2:
        return None

    best_break: Optional[int] = None
    best_diff: Optional[int] = None
    best_tokens: Optional[Tuple[int, int]] = None

    for k in range(1, len(rows)):
        left_rows = rows[:k]
        right_rows = rows[k:]
        if not right_rows:
            continue

        left_tokens = _chunk_token_count(header, left_rows)
        right_tokens = _chunk_token_count(header, right_rows)

        if left_tokens > token_limit or right_tokens > token_limit:
            continue

        diff = abs(left_tokens - right_tokens)
        if best_diff is None or diff < best_diff:
            best_break = k
            best_diff = diff
            best_tokens = (left_tokens, right_tokens)

    if best_break is None or best_tokens is None:
        return None

    left_rows = rows[:best_break]
    right_rows = rows[best_break:]
    chunks = [_build_chunk_lines(header, left_rows), _build_chunk_lines(header, right_rows)]
    stats = [
        ChunkStat(rows=len(left_rows), tokens=best_tokens[0]),
        ChunkStat(rows=len(right_rows), tokens=best_tokens[1]),
    ]
    return chunks, stats


def split_table(header: str, rows: Sequence[str], token_limit: int) -> tuple[list[list[str]], List[ChunkStat]]:
    """Greedy splitter that honors the provided token limit per chunk."""
    chunks: list[list[str]] = []
    stats: List[ChunkStat] = []
    current_rows: list[str] = []

    def flush_chunk(flush_rows: Sequence[str], overflow: bool = False) -> None:
        chunks.append(_build_chunk_lines(header, flush_rows))
        stats.append(ChunkStat(rows=len(flush_rows), tokens=_chunk_token_count(header, flush_rows), overflow=overflow))

    if not rows:
        flush_chunk(())
        return chunks, stats

    for row in rows:
        candidate_rows = current_rows + [row]
        candidate_tokens = _chunk_token_count(header, candidate_rows)
        if candidate_tokens <= token_limit:
            current_rows.append(row)
            continue

        if current_rows:
            flush_chunk(current_rows)
            current_rows = [row]
        else:
            flush_chunk([row], overflow=True)
            current_rows = []

    if current_rows:
        flush_chunk(current_rows)

    return chunks, stats


def process_tables(lines: List[str], max_tokens: int) -> tuple[List[str], List[TableStat]]:
    """Return new markdown lines with chunked tables and per-table stats."""
    output_lines: List[str] = []
    stats: List[TableStat] = []
    i = 0
    table_idx = 0
    total_lines = len(lines)

    while i < total_lines:
        line = lines[i]
        if line.strip() != TABLE_MARKER:
            output_lines.append(line)
            i += 1
            continue

        table_idx += 1
        start_line = i + 1  # 1-based line number for the marker
        if i + 1 >= total_lines:
            output_lines.append(line)
            break

        header_line = lines[i + 1]
        i += 2
        row_lines: list[str] = []
        while i < total_lines:
            stripped = lines[i].strip()
            if stripped == "" or stripped.startswith("#") or stripped == TABLE_MARKER:
                break
            row_lines.append(lines[i])
            i += 1

        original_tokens = _chunk_token_count(header_line, row_lines)

        chunk_blocks: List[List[str]]
        chunk_stats: List[ChunkStat]

        balanced_result: Optional[Tuple[List[List[str]], List[ChunkStat]]] = None
        if original_tokens > max_tokens and original_tokens < BALANCED_SPLIT_THRESHOLD:
            balanced_result = split_table_balanced(header_line, row_lines, token_limit=max_tokens)

        if balanced_result:
            chunk_blocks, chunk_stats = balanced_result
        else:
            chunk_limit = max_tokens
            if original_tokens > max_tokens and original_tokens < BALANCED_SPLIT_THRESHOLD:
                chunk_limit = max(1, math.ceil(original_tokens * BALANCED_SPLIT_RATIO))
            chunk_blocks, chunk_stats = split_table(header_line, row_lines, chunk_limit)

        stats.append(
            TableStat(
                index=table_idx,
                start_line=start_line,
                original_tokens=original_tokens,
                original_rows=len(row_lines),
                chunks=chunk_stats,
            )
        )

        for block_idx, block in enumerate(chunk_blocks):
            output_lines.extend(block)
            if block_idx != len(chunk_blocks) - 1:
                output_lines.append("")

    return output_lines, stats


def format_detailed_stats(table_stats: Iterable[TableStat]) -> str:
    """Return per-table chunk listings (always emitted)."""
    lines: list[str] = []
    for stat in table_stats:
        lines.append(
            f"Table {stat.index:03d}: rows={stat.original_rows} tokens={stat.original_tokens} -> {len(stat.chunks)} chunk(s)"
        )
        for idx, chunk in enumerate(stat.chunks, start=1):
            suffix = " (overflow)" if chunk.overflow else ""
            lines.append(f"    chunk {idx}: rows={chunk.rows} tokens={chunk.tokens}{suffix}")
    return "\n".join(lines)


def format_summary_stats(table_stats: Iterable[TableStat], max_tokens: int) -> str:
    lines: list[str] = []
    for stat in table_stats:
        chunk_count = len(stat.chunks)
        has_overflow = any(chunk.overflow for chunk in stat.chunks)
        if chunk_count == 1 and not has_overflow and stat.original_tokens <= max_tokens:
            continue

        chunk_sizes = ", ".join(str(chunk.tokens) for chunk in stat.chunks)
        lines.append(
            f"Found big table of {stat.original_tokens} tokens at line {stat.start_line} and created {chunk_count} tables in these sizes: {chunk_sizes}"
            + (" (overflow row)" if has_overflow and chunk_count == 1 else "")
        )

    if not lines:
        return "No tables exceeded the token limit."

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split CSV-style markdown tables into <=N token chunks")
    parser.add_argument("input", type=Path, help="Markdown file with CSV tables")
    parser.add_argument("--output", type=Path, help="Destination file (default: <input>_tablechunks.md)")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens per table chunk (default: {DEFAULT_MAX_TOKENS})",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    new_lines, table_stats = process_tables(lines, args.max_tokens)

    output_path = args.output or input_path.with_name(f"{input_path.stem}_tablechunks{input_path.suffix}")
    trailing_newline = "\n" if text.endswith("\n") else ""
    output_path.write_text("\n".join(new_lines) + trailing_newline, encoding="utf-8")

    print(format_detailed_stats(table_stats))
    print()
    print(format_summary_stats(table_stats, args.max_tokens))
    print(f"\nTables processed: {len(table_stats)}")
    print(f"Output written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
