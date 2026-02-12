from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

DEFAULT_FILTER_FIELDS = ["chunk_summary", "content", "heading", "concat_header_path"]


def load_chunks(config: Mapping[str, Any], *, logger: logging.Logger) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load the chunk corpus once and expose both an ordered list and a lookup map."""

    chunks_file = config.get("chunks_file") if config else None
    if not chunks_file:
        raise ValueError("Retrieval config must specify a 'chunks_file' entry")
    chunks_path = Path(chunks_file)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk corpus not found: {chunks_path}")

    with open(chunks_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    chunks = data.get("chunks") if isinstance(data, dict) and "chunks" in data else data
    if not isinstance(chunks, list):
        raise ValueError(f"Expected list of chunk records in {chunks_path}")

    ordered_chunks: List[Dict[str, Any]] = []
    chunk_map: Dict[str, Dict[str, Any]] = {}

    for chunk in chunks:
        chunk_id = chunk.get("id")
        content = (chunk.get("chunk_summary") or chunk.get("content") or "").strip()
        if not chunk_id or not content:
            continue
        key = str(chunk_id)
        ordered_chunks.append({"id": key, "text": content, "raw": chunk})
        chunk_map[key] = chunk

    if not ordered_chunks:
        raise ValueError(f"No usable chunks found in {chunks_path}")

    logger.info("Loaded %s candidate chunks from %s", len(ordered_chunks), chunks_path)
    return ordered_chunks, chunk_map


def get_chunk_field_value(chunk: Dict[str, Any], field: str) -> str:
    field = (field or "").strip().lower()
    if field in {"heading"}:
        return str(chunk.get("heading", ""))
    if field in {"path", "concat_header_path", "header_path"}:
        return str(chunk.get("concat_header_path", ""))
    if field in {"title"}:
        return str(chunk.get("title", ""))
    if field in {"filename", "file"}:
        return str(chunk.get("filename", ""))
    if field in {"category"}:
        return str(chunk.get("category", ""))
    return str(chunk.get(field, ""))


__all__ = ["DEFAULT_FILTER_FIELDS", "get_chunk_field_value", "load_chunks"]
