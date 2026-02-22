"""Detect duplicate headings that share the same parent path in a markdown file."""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
from typing import Dict, List, Tuple

HeadingKey = Tuple[Tuple[str, ...], str, int]
HeadingHits = Dict[HeadingKey, List[int]]
HEADING_RE = re.compile(r"^(#+)\s+(.*\S)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find headings that repeat under the same parent hierarchy"
    )
    parser.add_argument(
        "path",
        type=pathlib.Path,
        help="Markdown file to scan",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional report file path (default: <input dir>/duplicate_headings_same_parent.txt)",
    )
    return parser.parse_args()


def load_lines(path: pathlib.Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def find_duplicates(lines: List[str]) -> List[Tuple[Tuple[str, ...], str, int, List[int]]]:
    stack: List[Tuple[int, str]] = []
    hits: HeadingHits = collections.defaultdict(list)

    for idx, line in enumerate(lines, 1):
        match = HEADING_RE.match(line)
        if not match:
            continue
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent_path = tuple(item[1] for item in stack)
        hits[(parent_path, heading_text, level)].append(idx)
        stack.append((level, heading_text))

    duplicates = [
        (parent, heading, level, locations)
        for (parent, heading, level), locations in hits.items()
        if len(locations) > 1
    ]
    duplicates.sort(key=lambda item: (item[0], item[2], item[1]))
    return duplicates


def build_report(duplicates: List[Tuple[Tuple[str, ...], str, int, List[int]]]) -> str:
    lines = [f"Duplicate count: {len(duplicates)}"]
    for parent, heading, level, locations in duplicates:
        parent_str = " > ".join(parent) if parent else "(root)"
        lines.append(f"Parent: {parent_str}")
        locs = ", ".join(str(num) for num in locations)
        lines.append(f"Level {level} heading '{heading}' -> lines {locs}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    target = args.path.resolve()
    if not target.exists():
        raise SystemExit(f"File not found: {target}")

    duplicates = find_duplicates(load_lines(target))
    report = build_report(duplicates)

    if duplicates:
        print(f"Found {len(duplicates)} duplicate parent groups.")
    else:
        print("No duplicate headings share the same parent.")

    output_path = args.output
    if output_path is None:
        output_path = target.with_name("duplicate_headings_same_parent.txt")
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
