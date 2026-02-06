from __future__ import annotations
from pathlib import Path
import re
import sys
from io import StringIO
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

# Finds any inline HTML <table>...</table> blocks
TABLE_REGEX = re.compile(r"<table\b.*?>.*?</table>", re.IGNORECASE | re.DOTALL)

def _html_table_to_csv_block(table_html: str) -> str:
    """Convert one HTML <table> to a fenced CSV code block while preserving special chars in cells."""
    soup = BeautifulSoup(table_html, "html.parser")
    caption = soup.find("caption")
    caption_text = caption.get_text(strip=True) if caption else None

    # Extract table data using regex to preserve angle brackets and other special chars
    rows = []
    
    # Find all rows in the table using the original HTML
    tr_pattern = re.compile(r'<tr\b[^>]*>(.*?)</tr>', re.IGNORECASE | re.DOTALL)
    cell_pattern = re.compile(r'<(td|th)\b[^>]*>(.*?)</\1>', re.IGNORECASE | re.DOTALL)
    
    for tr_match in tr_pattern.finditer(table_html):
        row_html = tr_match.group(1)
        cells = []
        
        for cell_match in cell_pattern.finditer(row_html):
            cell_content = cell_match.group(2)
            # Keep all content, only clean up whitespace
            # This preserves < > \ / and other special characters
            cell_text = re.sub(r'\s+', ' ', cell_content.strip())
            cells.append(cell_text)
        
        if cells:  # Only add non-empty rows
            rows.append(cells)
    
    if not rows:
        return table_html  # fail-safe: leave original HTML
    
    # Convert to DataFrame to use pandas CSV formatting
    # Find the maximum number of columns
    max_cols = max(len(row) for row in rows) if rows else 0
    
    # Pad all rows to have the same number of columns
    for row in rows:
        while len(row) < max_cols:
            row.append("")
    
    # Create DataFrame
    df = pd.DataFrame(rows[1:] if len(rows) > 1 else [], 
                     columns=rows[0] if rows else [])

    # If no header row detected, use all rows as data
    if len(rows) == 1:
        df = pd.DataFrame([rows[0]])

    # Force \n line endings to avoid double-spacing in Markdown on Windows  
    csv_str = df.to_csv(index=False, lineterminator="\n").strip()

    note = f"Data Table"
    heading = f"Table: {caption_text}\n" if caption_text else ""
    # No leading/trailing blank lines; compact fenced block
    return f"<!-- {note} -->\n{heading}\n{csv_str}\n"

def convert_html_tables_to_csv(md_path: Path) -> Path:
    """
    Read Markdown, replace inline HTML tables with CSV code blocks,
    and write to <stem>_fixed.md in the same directory.
    """
    md_text = md_path.read_text(encoding="utf-8")

    def repl(match: re.Match) -> str:
        html = match.group(0)
        try:
            return _html_table_to_csv_block(html)
        except Exception:
            return html  # fail-safe: keep original table if anything goes wrong

    new_text = TABLE_REGEX.sub(repl, md_text)

    out_path = md_path.with_name(f"{md_path.stem}_withCSVtables.md")
    out_path.write_text(new_text, encoding="utf-8")
    return out_path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_tables.py <markdown_file>")
        sys.exit(1)

    md_file = Path(sys.argv[1])
    if not md_file.exists():
        print(f"Error: file not found: {md_file}")
        sys.exit(1)

    out = convert_html_tables_to_csv(md_file)
    print(f"✅ Wrote: {out}")
