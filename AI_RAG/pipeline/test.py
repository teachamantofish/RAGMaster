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

Empty headings should be chunked. However, "embedding" should be set to "false" so tahat the embedding
script skips them. This allows us to retain the document structure without bloating the vector DB with
empty chunks. We also preserver the concat_header_path so that the UI can display the full context.
"""

import os
import re
import uuid
import json
import yaml
import csv
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict
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
    logger.info(f"Loading markdown from {CWD}")

    # 1) Read docs from CWD (limit to markdown extensions)
    docs: List[Document] = SimpleDirectoryReader(
        str(CWD),
        recursive=True,
        required_exts=[".md", ".markdown", ".mdx"],   # <- add this
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

    for meta, text in linear_nodes:
        # filename still comes from metadata (path resolution unchanged)
        filename = _extract_filename(meta)
        if not filename:
            filename = "unknown.md"

        # Get / cache front-matter per file (parse once)
        if filename not in front_matter_by_file:
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    raw = f.read()
            except Exception:
                raw = text  # fallback to node text (may not include FM)
            fm, _ = parse_front_matter_text(raw)
            front_matter_by_file[filename] = fm

        # maintain stack for this file
        if filename not in stacks:
            stacks[filename] = []

        stack = stacks[filename]

        # Normalize body once
        body = (text or "").strip()
        if not body:
            continue

        # --- REQUIREMENT: derive heading/level from Markdown text, not metadata ---
        # If the first line is a Markdown ATX heading (e.g., "# Title", "## Subtitle", ...),
        # extract (level, heading) and treat the remainder as the node body.
        first_line, rest = (body.split("\n", 1) + [""])[:2]
        m = re.match(r"^(#{1,6})\s+(.+)$", first_line)
        if m:
            level = len(m.group(1))
            heading = m.group(2).strip()
            body = rest.strip()  # remainder becomes the body for this node
            # Update heading stack if this node has a heading level ≤ 6
            is_heading = (1 <= level <= 6 and heading != "")

            if is_heading:
                # pop to parent lower than this level
                while stack and stack[-1][1] >= level:
                    stack.pop()
                node_id = _new_id(f"h{level}")
                stack.append((heading, level, node_id))

                concat = " > ".join([s[0] for s in stack])
                parent_id = stack[-2][2] if len(stack) >= 2 else None
                if not body:
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
                        # id_prev / id_next left as default None

                        # heading / structure
                        heading=heading,
                        header_level=level,
                        concat_header_path=concat,

                        # content
                        content=body,
                        examples=[],

                        # summaries / metadata
                        # (defaults are fine: chunk_summary=None, page_summary=None, language=None)

                        # metrics / vectors
                        token_count=_tok(body) if body else 0,
                        # embedding left as default None
                    )
                )
            else:
                # Not expected with 1..6 constraint, but keep the branch for completeness.
                continue
        else:
            # Body-only node: attach to the current heading (no synthetic roots)
            if not stack:
                # We assume all text lives under headings; skip if stack is empty.
                continue
            concat = " > ".join([s[0] for s in stack])
            parent_id = stack[-1][2] if len(stack) >= 1 else None
            node_id = _new_id("leaf")

            candidates.append(
                LeafChunk(
                    # identity / linkage
                    id=node_id,
                    filename=filename,
                    parent_id=parent_id,
                    # id_prev / id_next left as default None

                    # heading / structure
                    heading=stack[-1][0],
                    header_level=stack[-1][1],
                    concat_header_path=concat,

                    # content
                    content=body,
                    examples=[],

                    # summaries / metadata
                    # (defaults are fine)

                    # metrics / vectors
                    token_count=_tok(body),
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

def _make_component_chunk(source: LeafChunk, *, content: str, chunk_type: str, language: Optional[str] = None) -> LeafChunk:
    """Create a chunk derived from ``source`` that holds a peeled component."""
    normalized = content.strip()
    return LeafChunk(
        id=_new_id(chunk_type[:3] if chunk_type else "cmp"),
        filename=source.filename,
        parent_id=source.parent_id,
        heading=source.heading,
        header_level=source.header_level,
        concat_header_path=source.concat_header_path,
        content=normalized,
        examples=[],
        chunk_type=chunk_type,
        chunk_summary=None,
        page_summary=None,
        language=language or source.language,
        token_count=_tok(normalized),
        embedding=None,
    )


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

    TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
    TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")

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

def enforce_chunk_size(chunks: List[LeafChunk]) -> List[LeafChunk]:
    """Ensure chunks respect ``MAX_TOKENS_FOR_NODE`` by peeling components."""
    if not ENABLE_CODE_EXTRACTION:
        logger.info("Component extraction disabled by config; skipping chunk size enforcement.")
        return chunks

    final_chunks: List[LeafChunk] = []

    for chunk in chunks:
        text = (chunk.content or "").strip()
        chunk.content = text
        chunk.chunk_type = "heading"
        chunk.token_count = _tok(text)
        chunk.examples = []

        components: List[LeafChunk] = []

        while chunk.token_count > MAX_TOKENS_FOR_NODE:
            # Pass 1: bleed off the largest fenced code block (usually examples/snippets).
            blocks = _find_code_blocks(text)
            largest_code = None
            largest_tokens = -1
            for block in blocks:
                block_text = text[block["start"]:block["end"]]
                tokens = _tok(block_text)
                if tokens > largest_tokens:
                    largest_tokens = tokens
                    largest_code = block

            if largest_code:
                code_text = largest_code["code"].rstrip()
                lang = _guess_lang(code_text, largest_code["lang"])
                fenced = f"```{largest_code['lang'] or ''}\n{code_text}\n```".strip()
                example_chunk = _make_component_chunk(chunk, content=fenced, chunk_type="example", language=lang)
                components.append(example_chunk)
                if example_chunk.id not in chunk.examples:
                    chunk.examples.append(example_chunk.id)
                text = (text[:largest_code["start"]] + text[largest_code["end"]:]).strip()
                chunk.content = text
                chunk.token_count = _tok(text)
                continue

            # Pass 2: if code peeling could not shrink enough, attempt the largest table block.
            tables = _find_tables(text)
            largest_table = None
            largest_tokens = -1
            for table in tables:
                table_text = text[table["start"]:table["end"]]
                tokens = _tok(table_text)
                if tokens > largest_tokens:
                    largest_tokens = tokens
                    largest_table = table

            if largest_table:
                table_body = text[largest_table["start"]:largest_table["end"]].strip()
                components.append(_make_component_chunk(chunk, content=table_body, chunk_type="table"))
                text = (text[:largest_table["start"]] + text[largest_table["end"]:]).strip()
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
        if not chunk.content:
            logger.error(
                "Chunk %s has no remaining content after component extraction",
                chunk.id,
                extra={"Parent Page": chunk.filename, "Token Count": chunk.token_count},
            )
        final_chunks.append(chunk)
        final_chunks.extend(components)

    logger.info(f"Chunks after size enforcement: {len(final_chunks)}")
    return final_chunks

# ----------------- prev/next -----------------
def link_prev_next(chunks: List[LeafChunk]) -> None:
    """Populate ``id_prev``/``id_next`` pointers so downstream UIs can paginate."""
    for i, ch in enumerate(chunks):
        ch.id_prev = chunks[i-1].id if i > 0 else None
        ch.id_next = chunks[i+1].id if i < len(chunks)-1 else None

# ----------------- Save (your schema) -----------------
def save_chunks_with_ordered_fields(chunks: List[dict], path: str, metadata: Dict):
    """
    Persist a list of chunk dicts to JSON with a stable field order.

    Field ordering mirrors the LeafChunk dataclass grouping:
      1) identity / linkage
      2) heading / structure
      3) content
      4) summaries / metadata
      5) metrics / vectors
      6) provenance (pipeline settings, models, etc.)

    Args:
        chunks: List of chunk dictionaries (already flattened, not dataclass instances).
        path:   Output JSON path.
        metadata: File-level front-matter dict; stamps standard keys
    """
    logger.info(f"Preparing to save {len(chunks)} chunks to {path}")
    if chunks:
        logger.info(f"First 5 chunk IDs: {[c.get('id') for c in chunks[:5]]}")

    ordered_chunks: List[OrderedDict] = []

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
        ordered["prov_id"] = f"prov_{datetime.now().strftime('%m_%d_%y')}"

        ordered_chunks.append(ordered)

    # Build provenance block once
    # Store provenance data as a chunk and push to a separate db table.
    # The chunk references the provenance data via prov_id
    provenance = {
        "prov_id": f"prov_{datetime.now().strftime('%m_%d_%y')}",
        "provenance": {
            "chunk": {
                "model": CHUNK_MODEL,
                "chunk_size_range": CHUNK_SIZE_RANGE,
                "keyword_density": KEYWORD_DENSITY,
            },
            "summary": {
                "model": CHUNK_SUMMARY_MODEL,
                "size": CHUNK_SUMMARY_SIZE,
                "temperature": CHUNK_SUMMARY_TEMPERATURE,
            },
            "embed": {
                "model": EMBED_MODEL,
                "vectorsize": VECTOR_SIZE,
            },
        },
    }

    # Final object: chunks + provenance
    # NOTE: we intentionally emit a wrapper object with a top-level
    # "chunks" key and a separate "provenance" block. This keeps file-level
    # provenance metadata (models, prompts, sizes, etc.) alongside the
    # chunk list so downstream scripts can reference the provenance without
    # duplicating it into every chunk. Consumers should therefore accept both
    # a bare list and this wrapper shape. When updating the file, preserve the
    # wrapper so provenance is not lost.
    out = {
        "chunks": ordered_chunks,
        "provenance": provenance
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

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

# TODO maybe: Add a quick schema validator (assert required keys present, types sane) 
# before writing the JSON, so a bad chunk can’t silently make it to disk.
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