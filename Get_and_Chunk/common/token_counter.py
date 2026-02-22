#!/usr/bin/env python3
"""
Enhanced Token Counter by Heading

Counts tokens for each heading section in a Markdown file using tiktoken
for accurate LLM token counting, with multiple formatting options.

Usage: pyt    parser.add_argument('--depth', type=int, default=6, choices=range(1, 7),
                       help='Maximum heading depth to analyze (1-6)')n count_tokens_enhanced.py <markdown_file> [options]

Options:
  --min-tokens N                      Only show headings with N+ tokens
  --format {tree,flat,summary,csv}    Output format (default: tree)
  --depth N                          Maximum heading depth to show (2-6)
"""

import sys
import re
import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import tiktoken

# Default configuration values (override here as needed)
OUTPUT_FORMAT = "csv"  # Options: "tree", "flat", "summary", "csv"
MIN_TOKENS = 0
MAX_HEADING_DEPTH = 6

def count_tokens(text: str) -> int:
    """
    Count tokens using tiktoken for accurate LLM token counting.
    Uses cl100k_base encoding (GPT-4/GPT-3.5-turbo tokenizer).
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

def parse_markdown_sections(content: str, max_depth: int = 6) -> List[Tuple[int, str, str]]:
    """Parse markdown content and return sections."""
    lines = content.split('\n')
    sections = []
    current_section = None
    current_content = []
    in_code_block = False
    
    for line in lines:
        # Check for code block markers
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            # Add line to current section content
            if current_section:
                current_content.append(line)
            continue
            
        # Skip heading detection if we're inside a code block
        if in_code_block:
            if current_section:
                current_content.append(line)
            continue
            
        # Check if line is a markdown heading (levels 1-6)
        # Allow one or more spaces between the hashes and the title to accommodate
        # headings that use additional padding for alignment.
        stripped_line = line.strip()
        heading_match = re.match(r'^(#{1,6})\s+(.*\S)$', stripped_line)
        
        if heading_match:
            level = len(heading_match.group(1))
            
            # Skip if beyond max depth
            if level > max_depth:
                continue
                
            # Save previous section if it exists
            if current_section:
                prev_level, prev_title = current_section
                content_text = '\n'.join(current_content).strip()
                sections.append((prev_level, prev_title, content_text))
            
            # Start new section
            title = heading_match.group(2).strip()
            current_section = (level, title)
            current_content = []
        else:
            # Add line to current section content
            if current_section:
                current_content.append(line)
    
    # Don't forget the last section
    if current_section:
        level, title = current_section
        content_text = '\n'.join(current_content).strip()
        sections.append((level, title, content_text))
    
    return sections


def print_tree_format(sections: List[Tuple[int, str, str]], token_func, min_tokens: int = 0):
    """Print token counts in hierarchical tree format."""
    total_tokens = 0
    shown_sections = 0
    
    for level, title, content in sections:
        token_count = token_func(content)
        total_tokens += token_count
        
        if token_count >= min_tokens:
            # Create indentation based on heading level
            indent = "  " * (level - 2)  # Start from level 2 (##)
            heading_markers = "#" * level
            
            print(f"{indent}{heading_markers} {title}")
            print(f"{indent}  {token_count}")
            shown_sections += 1
    
    print(f"\nSummary:")
    print(f"  Sections shown: {shown_sections}")
    print(f"  Total sections: {len(sections)}")
    print(f"  Total tokens: {total_tokens}")


def print_flat_format(sections: List[Tuple[int, str, str]], token_func, min_tokens: int = 0):
    """Print token counts in flat format."""
    results = []
    total_tokens = 0
    
    for level, title, content in sections:
        token_count = token_func(content)
        total_tokens += token_count
        
        if token_count >= min_tokens:
            heading_markers = "#" * level
            results.append((token_count, f"{heading_markers} {title}"))
    
    # Sort by token count (descending)
    results.sort(reverse=True)
    
    for token_count, heading in results:
        print(f"{token_count:6d}  {heading}")
    
    print(f"\nTotal tokens: {total_tokens}")


def _build_csv_text(sections: List[Tuple[int, str, str]], token_func, min_tokens: int = 0) -> Tuple[str, int]:
    """Return CSV string (header + rows) and total token count.

    The returned CSV does NOT include the trailing "# Total tokens" line; keep that for console only.
    """
    total_tokens = 0

    # First pass: calculate individual token counts
    section_data = []
    for level, title, content in sections:
        token_count = token_func(content)
        total_tokens += token_count

        if token_count >= min_tokens:
            heading_markers = "#" * level
            section_data.append({
                'md_header': heading_markers,
                'level': level,
                'header_text': title,
                'token_count': token_count,
                'cumulative': 0  # Will calculate in second pass
            })

    # Second pass: calculate cumulative counts for each section and its subsections
    for i, section in enumerate(section_data):
        current_level = section['level']
        has_children = False
        cumulative = section['token_count']  # Start with own tokens

        for j in range(i + 1, len(section_data)):
            next_section = section_data[j]
            if next_section['level'] <= current_level:
                break
            cumulative += next_section['token_count']
            has_children = True

        section['cumulative'] = cumulative if has_children else None

    # Build CSV text. Prepend a Row column with a sequential 1-based index.
    lines: List[str] = ["Row,MD header,Level,Header Text,Token #,Cumulative # count"]

    # Insert a summary/top row. Format (columns):
    # Row,MD header,Level,Header Text,Token #,Cumulative # count
    # We populate: Row="Total", MD header=<heading count>, Level=, Header Text=, Token #=, Cumulative # count=<total_tokens>
    heading_count = len(section_data)
    # heading_count placed in the MD header column to match requested layout
    # Ensure the summary row has the same number of columns as the header (6 columns).
    # Columns: Row,MD header,Level,Header Text,Token #,Cumulative # count
    lines.append(f'Total,{heading_count},,,,{total_tokens}')

    for idx, section in enumerate(section_data, start=1):
        header_text = section['header_text'].replace('"', '""')
        cumulative_str = str(section['cumulative']) if section['cumulative'] is not None else ''
        lines.append(f'{idx},"{section["md_header"]}",{section["level"]},"{header_text}",{section["token_count"]},{cumulative_str}')

    return "\n".join(lines) + "\n", total_tokens


def print_csv_format(sections: List[Tuple[int, str, str]], token_func, min_tokens: int = 0):
    """Print token counts in CSV format with cumulative counts."""
    csv_text, total_tokens = _build_csv_text(sections, token_func, min_tokens)
    print(csv_text, end='')
    print(f"\n# Total tokens: {total_tokens}")


def print_summary_format(sections: List[Tuple[int, str, str]], token_func, min_tokens: int = 0):
    """Print summary statistics by heading level."""
    level_stats = defaultdict(list)
    total_tokens = 0
    
    for level, title, content in sections:
        token_count = token_func(content)
        total_tokens += token_count
        
        if token_count >= min_tokens:
            level_stats[level].append(token_count)
    
    print("Summary by heading level:")
    print("-" * 40)
    
    for level in sorted(level_stats.keys()):
        counts = level_stats[level]
        heading_markers = "#" * level
        avg_tokens = sum(counts) / len(counts)
        print(f"{heading_markers} level:")
        print(f"  Sections: {len(counts)}")
        print(f"  Total tokens: {sum(counts)}")
        print(f"  Average tokens: {avg_tokens:.1f}")
        print(f"  Range: {min(counts)} - {max(counts)}")
        print()
    
    print(f"Overall total tokens: {total_tokens}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='Count tokens by heading using tiktoken for accurate LLM token counting')
    parser.add_argument('file', help='Markdown file to analyze')
    parser.add_argument('--min-tokens', type=int, default=MIN_TOKENS, 
                       help='Only show headings with this many tokens or more')
    parser.add_argument('--format', choices=['tree', 'flat', 'summary', 'csv'], 
                       default=OUTPUT_FORMAT, help='Output format')
    parser.add_argument('--depth', type=int, default=MAX_HEADING_DEPTH, choices=range(1, MAX_HEADING_DEPTH + 1),
                       help=f'Maximum heading depth to analyze (1-{MAX_HEADING_DEPTH})')
    parser.add_argument('--out', type=str, default=None,
                       help='Output file path when --format=csv. Defaults to <input_stem>_tokencount.csv next to the input file')
    
    args = parser.parse_args(argv)
    
    file_path = Path(args.file)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Outputs go next to the input file: CSV for --format=csv, .log otherwise
    
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Parse sections once
        sections = parse_markdown_sections(content, args.depth)

        if not sections:
            print("No headings found in the file.")
            return

        # CSV: write a real CSV file next to the input (or to --out)
        if args.format == 'csv':
            csv_text, total_tokens = _build_csv_text(sections, count_tokens, args.min_tokens)

            # Determine output path
            if args.out:
                out_path = Path(args.out).expanduser()
                if out_path.suffix == '':
                    out_path = out_path.with_suffix('.csv')
            else:
                out_path = file_path.parent / f"{file_path.stem}_markdown_tokencount.csv"

            out_path.parent.mkdir(parents=True, exist_ok=True)
            abs_out = out_path.resolve()
            abs_out.write_text(csv_text, encoding='utf-8')

            # Console output
            print(f"Token count analysis for: {file_path.name}")
            print(f"Using tiktoken (cl100k_base encoding), Min tokens: {args.min_tokens}, Format: csv")
            print("=" * 60)
            print(csv_text, end='')
            print(f"\n# Total tokens: {total_tokens}")
            print(f"\n📄 CSV written to: {abs_out}")

        else:
            # Non-CSV formats: capture and write a .log next to the input file
            import io
            from contextlib import redirect_stdout

            output_buffer = io.StringIO()

            with redirect_stdout(output_buffer):
                print(f"Token count analysis for: {file_path.name}")
                print(f"Using tiktoken (cl100k_base encoding), Min tokens: {args.min_tokens}, Format: {args.format}")
                print("=" * 60)

                if args.format == 'tree':
                    print_tree_format(sections, count_tokens, args.min_tokens)
                elif args.format == 'flat':
                    print_flat_format(sections, count_tokens, args.min_tokens)
                elif args.format == 'summary':
                    print_summary_format(sections, count_tokens, args.min_tokens)

            output_text = output_buffer.getvalue()

            # Write to console
            print(output_text, end='')

            # Log path next to input
            log_path = (file_path.parent / f"tokencount_{args.format}.log").resolve()
            with open(log_path, 'w', encoding='utf-8') as log_file:
                log_file.write(output_text)

            print(f"\n✅ Output written to: {log_path}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
