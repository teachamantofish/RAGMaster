"""Heading-first chunking script used by the pipeline.

This module walks LlamaIndex's Markdown nodes, rebuilds a document-level
heading hierarchy, and emits a JSON file of chunks sized for downstream
retrieval/embedding.  Oversized headings are trimmed by peeling code blocks
and tables into dedicated component chunks, and the script keeps track of
relationships such as prev/next pointers and provenance metadata.

## Principles

- Trust the upstream metadata (front matter + heading stack) and retain every heading as its own chunk.
- There is no "too small" merge pass—an empty heading is still emitted (and flagged in the logs).
- When a heading chunk is oversized, peel self-contained structures—code examples and markdown 
tables—into standalone chunks until the heading fits.
- Every emitted chunk keeps the same filename, parent id, and concatenated header path.

## Oversize handling

1. Measure the heading chunk against `MAX_TOKENS_FOR_NODE`.
2. While the chunk is over budget:
   - Remove the largest fenced code block, emit it as an `example` chunk, and recompute the heading size.
   - If no code blocks remain (or the chunk is still too large), remove the largest markdown table next, 
   emit it as a `table` chunk, and continue.
3. If the chunk is still too large after all candidates are exhausted, log a warning and leave it intact.
4. Log each emitted chunk with its type (`heading`, `example`, `table`) and token count.

## Undersize handling

Chunks under a certain threshold are handled in the summary phase. Short chunks have summaries
prepended to their content to provide more context during retrieval.

Empty headings should be chunked. However, "embedding" should be set to "false" so that the embedding
script skips them. This allows us to retain the document structure without bloating the vector DB with
empty chunks. We also preserver the concat_header_path so that the UI can display the full context.
"""

import os
import re
import uuid
import json
import yaml
import csv
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Deque
from datetime import datetime
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import MarkdownNodeParser
from pygments.lexers import guess_lexer, ClassNotFound

from config.chunkerconfig import *
from config.summaryconfig import *
from config.embedconfig import *
from common.utils import (get_csv_to_process, setup_global_logger)
from common.token_counter import main as run_token_counter

csvrow_data = get_csv_to_process() # Get the entire csv row to process, based dir, url, user metadata, etc. 
metadata = csvrow_data['input_csv_row'] # Store the row data in a var
CWD: Path = csvrow_data['cwd'] # Extract the rootdir/basedir from the csv row data
MD_TO_CHUNK: Path = CWD / f"{CWD.name}.md"
CHUNK_OUTPUT = CWD / OUTPUT_NAME

# Set up global loger with script-specific CSV header; overwrite existing log
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "Parent Page", "Token Count"]
logger = setup_global_logger(script_name=script_base, log_level='INFO', headers=LOG_HEADER)

# ----------------- Data -----------------
@dataclass
class LeafChunk:
    """In-memory representation of a chunk emitted by this pipeline."""
    # identity / linkage
    id: str
    filename: str
    parent_id: Optional[str]
    id_prev: Optional[str] = None
    id_next: Optional[str] = None

    # heading / structure
    heading: str = ""
    header_level: int = 0
    concat_header_path: str = ""

    # content
    content: str = ""
    examples: List[str] = field(default_factory=list)
    chunk_type: str = "heading"

    # summaries / metadata
    chunk_summary: Optional[str] = None
    page_summary: Optional[str] = None
    language: Optional[str] = None

    # metrics / vectors
    token_count: int = 0
    embedding: Optional[list] = None

# Use @dataclass default __repr__ for LeafChunk (keep representation simple)
# ----------------- Utilities -----------------
def _new_id(prefix: str = "n") -> str:
    """Generate a short, human-scannable identifier with the given prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _resolve_chunk_kind(kind: Optional[str], content: Optional[str]) -> str:
    """Normalize chunk kind based on explicit type or leading content markers."""
    normalized = (content or "").lstrip()
    detected = (kind or "heading").strip().lower()
    if detected == "heading":
        if normalized.startswith("```"):
            return "example"
        if normalized.startswith("<!-- Data Table -->"):
            return "table"
    return detected or "heading"


def build_chunk_id(header_level: int, *, chunk_type: str = "heading", content: Optional[str] = None) -> str:
    """Return an h#-prefixed chunk id and append _exa/_tab when needed."""
    # One place decides when IDs pick up the _exa/_tab suffix (either explicit chunk_type
    # or when heading content itself starts with a code fence / table marker).
    kind = _resolve_chunk_kind(chunk_type, content)
    suffix = ""
    if kind == "example":
        suffix = "_exa"
    elif kind == "table":
        suffix = "_tab"
    return f"{_new_id(f'h{header_level}')}" + suffix

# Define a global helper that all chunking code will use
def _tok(s: str) -> int:
    """Return token count using the global TOKENIZER."""
    if not s:
        return 0
    return len(TOKENIZER.encode(s))
# ----------------- Front-matter -----------------
FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

def parse_front_matter_text(md: str) -> Tuple[Dict, str]:
    m = FM_RE.match(md)
    if not m:
        return {}, md
    block = m.group(1)
    try:
        fm = yaml.safe_load(block) or {}
    except Exception:
        fm = {}
    body = md[m.end():]
    return fm, body


def _extract_heading_blocks(md_body: str) -> Deque[Tuple[str, int, str]]:
    """
    Parse markdown body (front matter removed) into ordered heading blocks.
    Returns deque of (heading_text, level, content_text).
    """
    blocks: List[Tuple[str, int, str]] = []
    current_heading: Optional[str] = None
    current_level: Optional[int] = None
    current_lines: List[str] = []

    in_fence = False
    fence_delim = None  # Track fenced blocks so comments with a # at the start of a line are not treated as headings. 

    for line in md_body.splitlines():
        stripped = line.strip()
        fence_match = re.match(r"^(`{3,}|~{3,})(.*)$", stripped)
        if fence_match:
            delim = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_delim = delim
            elif fence_delim == delim:
                in_fence = False
                fence_delim = None
            if current_heading is not None:
                current_lines.append(line)
            continue

        if in_fence:
            if current_heading is not None:
                current_lines.append(line)
            continue
        # We are not in a fenced block, so we can check for headings
        match = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if match:
            if current_heading is not None:
                blocks.append((current_heading, current_level or 0, "\n".join(current_lines).strip()))
            current_heading = match.group(2).strip()
            current_level = len(match.group(1))
            current_lines = []
        else:
            if current_heading is not None:
                current_lines.append(line)

    if current_heading is not None:
        blocks.append((current_heading, current_level or 0, "\n".join(current_lines).strip()))

    return deque(blocks)

# ----------------- LlamaIndex → linear node stream -----------------
def _extract_heading(meta: Dict) -> Tuple[str, int]:
    """
    Extract heading text/level from an ATX-style heading string (e.g., "## Title").
    Returns ("", 999) when no heading is present.
    """
    raw = str(meta.get("heading") or "").strip()
    match = re.match(r"^(#{1,6})\s+(.+)$", raw)
    if not match:
        return "", 999

    level = len(match.group(1))
    heading = match.group(2).strip()
    return heading, level


def _extract_filename(meta: Dict) -> str:
    # Accept only 'filename' or 'file_path' (no other fallbacks). Normalize to
    # a CWD-relative path when possible. Raise if both keys are missing.
    raw = meta.get("file_path", "")
    if not raw:
        raise ValueError("meta['file_path'] is missing or empty")

    p = Path(raw)
    try:
        return str(p.resolve(strict=False).relative_to(CWD.resolve())).replace("\\", "/")
    except Exception:
        # return the original file_path string (normalized slashes) as the fallback
        return str(raw).replace("\\", "/")
    

def load_llamaindex_nodes() -> List[Tuple[Dict, str]]:
    """
    Returns a linear list of (metadata, text) in document order across files.
    Each item comes from MarkdownNodeParser.get_nodes_from_documents().
    """
    if MD_TO_CHUNK.exists():
        logger.info(f"Loading markdown from single file {MD_TO_CHUNK}")
        docs: List[Document] = SimpleDirectoryReader(
            input_files=[str(MD_TO_CHUNK)],
        ).load_data()
    else:
        logger.info(f"Loading markdown from directory {CWD}")
        docs: List[Document] = SimpleDirectoryReader(
            str(CWD),
            recursive=True,
            required_exts=[".md"],
        ).load_data()
    logger.info(f"SimpleDirectoryReader loaded {len(docs)} files")

    # 2) Parse into nodes (Markdown-aware)
    md_parser = MarkdownNodeParser()
    nodes = md_parser.get_nodes_from_documents(docs)
    logger.info(f"MarkdownNodeParser produced {len(nodes)} nodes")

    # 3) Flatten to (metadata, text)
    linear: List[Tuple[Dict, str]] = []
    for n in nodes:
        meta = dict(n.metadata or {})
        text = n.get_content(metadata_mode="none")

        if "file_path" not in meta and "filename" not in meta:
            src = n.source_node
            if src and hasattr(src, "metadata"):
                srcm = dict(src.metadata)
                fp = _extract_filename(srcm)
                if fp:
                    meta["file_path"] = fp

        linear.append((meta, text))

    return linear

# ----------------- Build heading stack & propose leaves -----------------
def build_candidates_from_linear(linear_nodes: List[Tuple[Dict, str]]) -> Tuple[List[LeafChunk], Dict[str, Dict]]:
    """
    Reconstruct a heading stack while walking nodes in order.
    Emit *leaf candidates* (no children) as LeafChunk.
    Also returns a front_matter map per filename (parsed from doc-level text).
    """
    candidates: List[LeafChunk] = []
    # Track stacks per file to compute concat_header_path and parent_id
    stacks: Dict[str, List[Tuple[str, int, str]]] = {}  # filename -> [(heading, level, node_id)]
    # per-file front matter
    front_matter_by_file: Dict[str, Dict] = {}
    # cache ordered heading blocks extracted from the source markdown
    heading_blocks: Dict[str, Deque[Tuple[str, int, str]]] = {}

    for meta, text in linear_nodes:
        # filename still comes from metadata (path resolution unchanged)
        filename = _extract_filename(meta)
        if not filename:
            filename = "unknown.md"

        # Get / cache front-matter per file (parse once)
        if filename not in front_matter_by_file:
            try:
                file_path = Path(filename)
                if not file_path.is_absolute():
                    file_path = (CWD / file_path).resolve()
                with open(file_path, "r", encoding="utf-8") as f:
                    raw = f.read()
            except Exception:
                raw = text  # fallback to node text (may not include FM)
            fm, body_text = parse_front_matter_text(raw)
            front_matter_by_file[filename] = fm
            heading_blocks[filename] = _extract_heading_blocks(body_text)

        # maintain stack for this file
        if filename not in stacks:
            stacks[filename] = []

        stack = stacks[filename]

        blocks = heading_blocks.get(filename)
        if not blocks:
            continue

        # Pop the next heading/content block extracted from source markdown
        heading, level, body = blocks.popleft()
        body = (body or "").strip()
        if heading == "":
            continue

        # pop to parent lower than this level
        while stack and stack[-1][1] >= level:
            stack.pop()
        node_id = build_chunk_id(level, chunk_type="heading", content=body)
        stack.append((heading, level, node_id))

        concat = " > ".join([s[0] for s in stack])
        parent_id = stack[-2][2] if len(stack) >= 2 else None
        if not body:
            # Heading with no text, only subheadings: when a heading is immediately followed by
            # subheadings, _extract_heading_blocks yields an empty body for the parent. This function
            # still emits a chunk, logs “Empty heading chunk encountered,” and sets its token count to
            # zero (3chunker.py:242-337). Downstream, chunks_to_dicts sees the zero tokens and forces
            # embedding/summary flags to "false", so later stages skip embedding while keeping the
            # header path so the UI can show the empty node in context (3chunker.py:815-844). The child
            # subheadings, each with their own content, are chunked normally when their blocks are
            # processed.
            logger.error(
                "Empty heading chunk encountered: %s",
                heading,
                extra={"Parent Page": filename, "Token Count": 0},
            )

        candidates.append(
            LeafChunk(
                # identity / linkage
                id=node_id,
                filename=filename,
                parent_id=parent_id,
                # heading / structure
                heading=heading,
                header_level=level,
                concat_header_path=concat,
                # content
                content=body,
                examples=[],
                # metrics / vectors
                token_count=_tok(body) if body else 0,
            )
        )
    # Return the constructed candidates list and the per-file front-matter map
    return candidates, front_matter_by_file

# ----------------- Long-code extraction -----------------

# None of this runs if ENABLE_CODE_EXTRACTION is False. 

CODE_BLOCK_RE = re.compile(r"```([A-Za-z0-9_\-+.]*)\s*\n(.*?)\n```", re.DOTALL)

def _find_code_blocks(text: str) -> List[dict]:
    """Return start/end offsets for fenced code blocks within ``text``."""
    out = []
    for m in CODE_BLOCK_RE.finditer(text):
        out.append({"start": m.start(), "end": m.end(), "lang": m.group(1) or None, "code": m.group(2)})
    return out

def _guess_lang(code: str, fallback: Optional[str]) -> Optional[str]:
    """Prefer the explicit fence language, otherwise lean on Pygments heuristics."""
    if fallback:
        return fallback.strip()

    try:
        lx = guess_lexer(code)
        return lx.name
    except ClassNotFound:
        return None

def _split_code_block(code: str, lang: Optional[str]) -> List[str]:
    """Split ``code`` into logical sub-blocks when possible."""
    language = (lang or "").lower()
    if language in {"python", "py"} or re.search(r"(?m)^def\s+\w", code):
        return _split_python_functions(code)
    if language in {"javascript", "js", "typescript", "ts", "jsx", "tsx", "extendscript"} or re.search(
        r"(?m)^(?:export\s+)?(?:async\s+)?function\s+\w", code
    ):
        return _split_js_functions(code)
    return [code]

def _split_python_functions(code: str) -> List[str]:
    """Split top-level Python functions while keeping leading comments."""
    lines = code.splitlines()
    if not lines:
        return [code]

    starts: List[int] = []
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith("def ") and indent == 0:
            start = idx
            look = idx - 1
            while look >= 0:
                prev = lines[look]
                if not prev.strip():
                    break
                prev_stripped = prev.lstrip()
                if prev_stripped.startswith("#") and len(prev) - len(prev_stripped) == 0:
                    start = look
                    look -= 1
                    continue
                break
            starts.append(start)

    if not starts:
        return [code]

    starts.append(len(lines))
    segments: List[str] = []
    for left, right in zip(starts, starts[1:]):
        segment = "\n".join(lines[left:right]).strip("\n")
        if segment.strip():
            segments.append(segment)

    return segments or [code]

def _split_js_functions(code: str) -> List[str]:
    """Split top-level JavaScript/TypeScript-style functions, including leading comments."""
    lines = code.splitlines()
    if not lines:
        return [code]

    def is_function_start(text: str) -> bool:
        patterns = [
            r"(?:export\s+)?(?:async\s+)?function\s+\w+\s*\(",
            r"(?:export\s+)?default\s+function\s+\w*\s*\(",
            r"(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?function\s*\(",
            r"(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>\s*{",
        ]
        return any(re.match(p, text) for p in patterns)

    starts: List[int] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if is_function_start(stripped):
            start = idx
            look = idx - 1
            while look >= 0:
                prev = lines[look]
                prev_stripped = prev.strip()
                if not prev_stripped:
                    break
                if prev_stripped.startswith("//") or prev_stripped.startswith("/*") or prev_stripped.startswith("*"):
                    start = look
                    look -= 1
                    continue
                break
            starts.append(start)

    if not starts:
        return [code]

    starts.append(len(lines))
    segments: List[str] = []
    for left, right in zip(starts, starts[1:]):
        segment = "\n".join(lines[left:right]).strip("\n")
        if segment.strip():
            segments.append(segment)

    return segments or [code]

def _make_component_chunk(source: LeafChunk, *, content: str, chunk_type: str, language: Optional[str] = None) -> LeafChunk:
    """Create a chunk derived from ``source`` that holds a peeled component."""
    normalized = content.strip()
    resolved_kind = _resolve_chunk_kind(chunk_type, normalized)
    if resolved_kind == "table":
        # Replace legacy table markers with a descriptive title tied to the parent heading.
        normalized = _decorate_table_chunk(source.heading, normalized)
    chunk_id = build_chunk_id(source.header_level, chunk_type=resolved_kind, content=normalized)
    return LeafChunk(
        id=chunk_id,
        filename=source.filename,
        parent_id=source.parent_id,
        heading=source.heading,
        header_level=source.header_level,
        concat_header_path=source.concat_header_path,
        content=normalized,
        examples=[],
        chunk_type=resolved_kind,
        chunk_summary=None,
        page_summary=None,
        language=language or source.language,
        token_count=_tok(normalized),
        embedding=None,
    )


TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _find_tables(text: str) -> List[Dict[str, int]]:
    """Locate tabular regions (markdown pipe or CSV) outside of code fences."""
    tables: List[Dict[str, int]] = []
    if not text:
        return tables

    code_ranges = [(m.start(), m.end()) for m in CODE_BLOCK_RE.finditer(text)]

    def _in_code(idx: int) -> bool:
        return any(start <= idx < end for start, end in code_ranges)

    lines = text.splitlines(keepends=True)
    idx = 0
    block_start: Optional[int] = None
    block_type: Optional[str] = None  # "pipe" | "csv"
    expected_cols: Optional[int] = None
    block_lines = 0

    def count_header_columns(line: str) -> Optional[int]:
        if "," not in line:
            return None
        try:
            row = next(csv.reader([line]))
            if len(row) >= 2:
                return len(row)
        except Exception:
            pass

        # Fallback: simple comma count (header rows are usually clean).
        cols = sum(1 for _ in line.split(","))
        return cols if cols >= 2 else None

    def has_required_columns(line: str, expected: Optional[int]) -> bool:
        if expected is None or expected < 2:
            return False
        if "," not in line:
            return False
        parts = line.split(",", expected - 1)
        return len(parts) >= expected and any(p.strip() for p in parts[1:])

    def flush(end_idx: int) -> None:
        nonlocal block_start, block_type, expected_cols, block_lines
        if block_start is not None and block_lines >= 2:
            tables.append({"start": block_start, "end": end_idx})
        block_start = None
        block_type = None
        expected_cols = None
        block_lines = 0

    for line in lines:
        line_end = idx + len(line)

        if _in_code(idx):
            flush(idx)
            idx = line_end
            continue

        stripped = line.strip()
        if not stripped:
            flush(idx)
            idx = line_end
            continue

        is_pipe = bool(TABLE_ROW_RE.match(stripped))
        csv_candidate = not is_pipe and "," in stripped

        if is_pipe:
            if block_type not in ("pipe", None):
                flush(idx)
            if block_start is None:
                block_start = idx
                block_type = "pipe"
            block_lines += 1
        elif csv_candidate:
            if block_type not in ("csv", None):
                flush(idx)
            if block_type != "csv":
                header_cols = count_header_columns(stripped)
                if header_cols is None:
                    idx = line_end
                    continue
                block_start = idx
                block_type = "csv"
                expected_cols = header_cols
                block_lines = 1
            else:
                if expected_cols is None:
                    expected_cols = count_header_columns(stripped)
                if has_required_columns(stripped, expected_cols):
                    block_lines += 1
                else:
                    flush(idx)
                    header_cols = count_header_columns(stripped)
                    if header_cols is not None:
                        block_start = idx
                        block_type = "csv"
                        expected_cols = header_cols
                        block_lines = 1
                    else:
                        block_start = None
                        block_type = None
                        expected_cols = None
                        block_lines = 0
        else:
            flush(idx)

        idx = line_end

    flush(len(text))

    # Filter pipe tables that lack separator rows to avoid false positives.
    filtered: List[Dict[str, int]] = []
    for tb in tables:
        segment = text[tb["start"]:tb["end"]]
        if "|" in segment:
            if segment.count("\n") >= 1 and (TABLE_SEP_RE.search(segment) or "|---" in segment):
                filtered.append(tb)
        else:
            # CSV block - accept as-is
            filtered.append(tb)

    return filtered


def _expand_table_region(text: str, start_idx: int) -> int:
    """Extend a table slice upward to grab leading markers/blank lines."""
    expanded = start_idx
    while expanded > 0:
        prev_nl = text.rfind("\n", 0, max(expanded - 1, 0))
        line_start = 0 if prev_nl == -1 else prev_nl + 1
        candidate = text[line_start:expanded]
        stripped = candidate.strip()
        if not stripped:
            expanded = line_start
            continue
        if stripped == "<!-- Data Table -->" or stripped.startswith("Table:"):
            expanded = line_start
            continue
        break
    return expanded


def _strip_table_wrappers(table_text: str) -> str:
    """Drop legacy table markers/captions before rebuilding the title."""
    cleaned: List[str] = []
    for line in table_text.splitlines():
        stripped = line.strip()
        if not stripped and not cleaned:
            continue
        if stripped == "<!-- Data Table -->" or stripped.startswith("Table:"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _try_parse_csv_rows(table_body: str) -> List[List[str]]:
    lines = [ln for ln in table_body.splitlines() if ln.strip()]
    if not lines or not any("," in ln for ln in lines[:2]):
        return []
    try:
        reader = csv.reader(lines)
        rows = [[cell.strip() for cell in row] for row in reader]
    except Exception:
        return []
    if len(rows) == 1 and len(rows[0]) <= 1:
        return []
    return rows


def _try_parse_pipe_rows(table_body: str) -> List[List[str]]:
    lines = [ln for ln in table_body.splitlines() if ln.strip()]
    if not lines or not any("|" in ln for ln in lines):
        return []
    rows: List[List[str]] = []
    for line in lines:
        stripped = line.strip()
        if TABLE_SEP_RE.match(stripped):
            continue
        if "|" not in stripped:
            return []
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        rows.append(cells)
    return rows


def _summarize_table_rows(table_body: str) -> Optional[Dict[str, object]]:
    rows = _try_parse_csv_rows(table_body)
    header_offset = 0
    if rows:
        header_offset = 1 if len(rows) > 1 else 0
    else:
        rows = _try_parse_pipe_rows(table_body)
        header_offset = 1 if rows and len(rows) > 1 else 0
    if not rows:
        return None
    data_rows = rows[header_offset:] if len(rows) > header_offset else rows
    data_rows = [row for row in data_rows if row and any(cell.strip() for cell in row)]
    if not data_rows:
        return None
    first_value = (data_rows[0][0].strip() if data_rows[0] and data_rows[0][0].strip() else "n/a")
    last_value = (data_rows[-1][0].strip() if data_rows[-1] and data_rows[-1][0].strip() else "n/a")
    return {
        "first_value": first_value or "n/a",
        "last_value": last_value or "n/a",
        "row_count": len(data_rows),
    }


def _decorate_table_chunk(heading: str, table_text: str) -> str:
    """Inject a descriptive title using the parent heading and table bounds."""
    focus_heading = heading.strip() or "Untitled Table"
    core = _strip_table_wrappers(table_text)
    if not core:
        return f"Table: {focus_heading} (from n/a to n/a: 0)"
    summary = _summarize_table_rows(core)
    if summary:
        start_val = summary.get("first_value", "n/a") or "n/a"
        end_val = summary.get("last_value", "n/a") or "n/a"
        row_count = summary.get("row_count", 0) or 0
    else:
        start_val = "n/a"
        end_val = "n/a"
        row_count = 0
    title = f"Table: {focus_heading} (from {start_val} to {end_val}: {row_count} rows)"
    return f"{title}\n\n{core}" if core else title

def enforce_chunk_size(chunks: List[LeafChunk]) -> List[LeafChunk]:
    """Ensure chunks respect ``MAX_TOKENS_FOR_NODE`` by peeling components."""
    if not ENABLE_CODE_EXTRACTION:
        logger.info("Component extraction disabled by config; skipping chunk size enforcement.")
        return chunks

    final_chunks: List[LeafChunk] = []

    for chunk in chunks:
        text = (chunk.content or "").strip()
        chunk.content = text
        chunk.token_count = _tok(text)
        chunk.examples = []

        components: List[LeafChunk] = []

        while chunk.token_count > MAX_TOKENS_FOR_NODE:
            # Pass 1: bleed off the largest fenced code block (usually examples/snippets).
            blocks = _find_code_blocks(text)
            # Identify the largest fenced code block within the current chunk.
            largest_code = None
            largest_tokens = -1
            for block in blocks:
                block_text = text[block["start"]:block["end"]]
                tokens = _tok(block_text)
                if tokens > largest_tokens:
                    largest_tokens = tokens
                    largest_code = block

            if largest_code:
                # Promote the largest fenced code block into its own example component chunk.
                code_text = largest_code["code"].rstrip()
                lang = _guess_lang(code_text, largest_code["lang"])
                sub_blocks = _split_code_block(code_text, lang or largest_code["lang"])
                for sub in sub_blocks:
                    fenced = f"```{largest_code['lang'] or ''}\n{sub.strip()}\n```".strip()
                    example_chunk = _make_component_chunk(chunk, content=fenced, chunk_type="example", language=lang)
                    components.append(example_chunk)
                    if example_chunk.id not in chunk.examples:
                        chunk.examples.append(example_chunk.id)
                # Excise the peeled code block and recalculate the parent chunk tokens.
                text = (text[:largest_code["start"]] + text[largest_code["end"]:]).strip()
                chunk.content = text
                chunk.token_count = _tok(text)
                continue

            # Pass 2: if code peeling could not shrink enough, attempt the last table block found.
            # Repeats until the size constraint is satisfied or no tables are left.
            tables = _find_tables(text)
            tail_table = tables[-1] if tables else None

            if tail_table:
                table_start = _expand_table_region(text, tail_table["start"])
                table_body = text[table_start:tail_table["end"]].strip()
                # Strip the marker/comment wrapper alongside the CSV so the parent chunk keeps only real prose.
                components.append(_make_component_chunk(chunk, content=table_body, chunk_type="table"))
                text = (text[:table_start] + text[tail_table["end"]:]).strip()
                chunk.content = text
                chunk.token_count = _tok(text)
                continue

            # No removable components remain and the chunk is still large.
            logger.warning(
                "Chunk %s remains over max size (%s tokens) despite removing components",
                chunk.id,
                chunk.token_count,
            )
            break

        chunk.content = text
        chunk.token_count = _tok(text)
        chunk.chunk_type = _resolve_chunk_kind(chunk.chunk_type, chunk.content)
        if chunk.chunk_type == "table":
            first_line = chunk.content.splitlines()[0].strip() if chunk.content else ""
            needs_title = "<!-- Data Table -->" in chunk.content or (first_line.startswith("Table:") and "(from" not in first_line)
            if needs_title:
                chunk.content = _decorate_table_chunk(chunk.heading or chunk.concat_header_path, chunk.content)
                chunk.token_count = _tok(chunk.content)
        if not chunk.content:
            # Heading with only a code example: the chunk is emitted as a normal heading chunk unless
            # it exceeds MAX_TOKENS_FOR_NODE. When it’s oversized, the largest fenced block is peeled
            # into its own chunk_type="example" chunk, and the parent heading shrinks accordingly
            # (3chunker.py:602-658). If that was the only content, the heading chunk becomes empty; the
            # script logs it, but still keeps the chunk so the hierarchy remains intact, and later marks
            # it non-embeddable (embedding="false") because its token count is zero. In short, the 
            # code example survives as a separate component chunk while the heading stub persists for structure.
            logger.info(
                "Chunk %s has 0 token count after moving child chunks to their own chunk due to MAX_TOKENS_FOR_NODE threshold.",
                chunk.id,
                extra={"Parent Page": chunk.filename, "Token Count": chunk.token_count},
            )
        final_chunks.append(chunk)
        final_chunks.extend(components)

    logger.info(f"Chunks after size enforcement: {len(final_chunks)}")
    return final_chunks

# ----------------- prev/next -----------------
def link_prev_next(chunks: List[LeafChunk]) -> None:
    """Populate ``id_prev``/``id_next`` to facility retreival and so pointers so downstream UIs can paginate."""
    for i, ch in enumerate(chunks):
        ch.id_prev = chunks[i-1].id if i > 0 else None
        ch.id_next = chunks[i+1].id if i < len(chunks)-1 else None

# ----------------- Save (your schema) -----------------
def save_chunks_with_ordered_fields(chunks: List[dict], path: str, metadata: Dict):
    """
    Persist a list of chunk dicts to JSON with a stable field order.
    Provenance is saved separately to a_provenance.json.

    Field ordering mirrors the LeafChunk dataclass grouping:
      1) identity / linkage
      2) heading / structure
      3) content
      4) summaries / metadata
      5) metrics / vectors

    Args:
        chunks: List of chunk dictionaries (already flattened, not dataclass instances).
        path:   Output JSON path.
        metadata: File-level front-matter dict; stamps standard keys
    """
    logger.info(f"Preparing to save {len(chunks)} chunks to {path}")
    if chunks:
        logger.info(f"First 5 chunk IDs: {[c.get('id') for c in chunks[:5]]}")

    ordered_chunks: List[OrderedDict] = []

    prov_id = f"prov_{datetime.now().strftime('%m_%d_%y')}"

    for chunk in chunks:
        ordered = OrderedDict()
        # --- 1) identity / linkage ---
        ordered["id"] = chunk.get("id")
        ordered["filename"] = chunk.get("filename")
        ordered["parent_id"] = chunk.get("parent_id")
        ordered["id_prev"] = chunk.get("id_prev")
        ordered["id_next"] = chunk.get("id_next")

        # --- 2) heading / structure ---
        ordered["heading"] = chunk.get("heading")
        ordered["header_level"] = chunk.get("header_level")
        ordered["concat_header_path"] = chunk.get("concat_header_path")
        ordered["chunk_type"] = chunk.get("chunk_type")

        # --- 3) content ---
        ordered["content"] = chunk.get("content")
        ordered["examples"] = chunk.get("examples")

        # --- 4) summaries / metadata ---
        ordered["chunk_summary"] = chunk.get("chunk_summary")
        ordered["page_summary"] = chunk.get("page_summary")
        ordered["title"] = metadata.get("METADATA_TITLE", "")
        ordered["author"] = metadata.get("METADATA_AUTHOR", "")
        ordered["category"] = metadata.get("METADATA_CATEGORY", "")
        ordered["description"] = metadata.get("METADATA_DESCRIPTION", "")
        ordered["language"] = chunk.get("language")

        # --- 5) metrics / vectors ---
        ordered["token_count"] = chunk.get("token_count")
        ordered["embedding"] = chunk.get("embedding")

        # --- 6) provenance reference ---
        ordered["prov_id"] = prov_id

        ordered_chunks.append(ordered)

    # Build provenance block once
    provenance = {
        "prov_id": prov_id,
        "timestamp": datetime.now().strftime('%m_%d_%y'),
        "chunk": {
            "model": CHUNK_MODEL,
            "chunk_size_range": CHUNK_SIZE_RANGE,
            "keyword_density": KEYWORD_DENSITY,
        },
        "summary": {
            "model": "",
            "prompt": "",
            "size": "",
            "temperature": "",
        },
        "embed": {
            "model": "",
            "vectorsize": "",
        },
    }

    # Write chunks as plain JSON array
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(ordered_chunks, f, indent=2, ensure_ascii=False)

    # Write simple CSV with the same token counts stored in JSON
    csv_path = Path(path).with_name("chunk_token_counts_report.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["header_level", "heading", "token_count"])
        for ordered in ordered_chunks:
            writer.writerow([
                ordered.get("header_level", ""),
                ordered.get("heading", ""),
                ordered.get("token_count", ""),
            ])

    # Write provenance separately
    prov_path = Path(path).parent / "a_provenance.json"
    with open(prov_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(ordered_chunks)} chunks to {path}")
    logger.info(f"Saved token counts CSV to {csv_path}")
    logger.info(f"Saved provenance to {prov_path}")

def chunks_to_dicts(chunks: List[LeafChunk]) -> List[dict]:
    """
    Convert LeafChunk dataclass instances to plain dicts (no embeddings computed here),
    keeping keys in the same logical groups/order used by save_chunks_with_ordered_fields.

    When the token count is 0, set embedding to "false" so that the summary and 
    embedding script skips it.
    """
    out: List[dict] = []
    for ch in chunks:
        embedding_value = ch.embedding
        chunk_summary_value = ch.chunk_summary
        page_summary_value = ch.page_summary
        if ch.token_count == 0:
            embedding_value = "false"
            chunk_summary_value = "false"
            page_summary_value = "false"
        out.append({
            # 1) identity / linkage
            "id": ch.id,
            "filename": ch.filename,
            "parent_id": ch.parent_id,
            "id_prev": ch.id_prev,
            "id_next": ch.id_next,

            # 2) heading / structure
            "heading": ch.heading,
            "header_level": ch.header_level,
            "concat_header_path": ch.concat_header_path,
            "chunk_type": ch.chunk_type,

            # 3) content
            "content": ch.content,
            "examples": ch.examples,

            # 4) summaries / metadata
            "chunk_summary": chunk_summary_value,
            "page_summary": page_summary_value,
            "language": ch.language,

            # 5) metrics / vectors
            "token_count": ch.token_count,
            "embedding": embedding_value,
        })
    return out

# ----------------- Driver -----------------
def process_directory_llamaindex():
    """
    1) LlamaIndex reads & parses markdown
    2) We reconstruct heading stacks → leaf candidates
    3) Enforce size limits by peeling code examples/tables into separate chunks
    4) Link prev/next pointers
    5) Emit ALL chunks to CWD/a_chunks.json
    """
    linear = load_llamaindex_nodes()

    candidates, fm_by_file = build_candidates_from_linear(linear)
    logger.info(f"Candidates built: {len(candidates)}")

    final_chunks = enforce_chunk_size(candidates)
    logger.info(f"Final chunk count: {len(final_chunks)}")

    link_prev_next(final_chunks)

    # Log a CSV row per chunk using the declared extra columns in LOG_HEADER.
    # The CSVFormatter in common.utils will place these extras into the
    # corresponding columns (e.g. "Parent Page", "Token Count").
    for ch in final_chunks:
        # Message column: include chunk id and a short heading for context
        msg = f"chunk:{ch.id} type={ch.chunk_type} heading={ch.heading[:80]}"
        try:
            logger.info(msg, extra={"Parent Page": ch.filename, "Token Count": ch.token_count})
        except Exception:
            # Ensure logging never breaks the pipeline; fall back to simple info
            logger.info(f"{msg} parent={ch.filename} tokens={ch.token_count}")

    save_chunks_with_ordered_fields(chunks_to_dicts(final_chunks), CHUNK_OUTPUT, metadata=metadata)

    logger.info(f"Done. Wrote {len(final_chunks)} chunks to {CHUNK_OUTPUT}")

# If run directly:
if __name__ == "__main__":
    process_directory_llamaindex()
    run_token_counter([str(CHUNK_OUTPUT)])
