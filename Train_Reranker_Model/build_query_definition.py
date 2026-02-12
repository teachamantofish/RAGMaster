from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from chunks_loader import DEFAULT_FILTER_FIELDS, get_chunk_field_value


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for value in values:
        if not value:
            continue
        key = str(value)
        if key in seen:
            continue
        deduped.append(key)
        seen.add(key)
    return deduped


def _matches_filter_spec(text: str, spec: Dict[str, Any]) -> bool:
    text_lower = text.lower()
    contains = str(spec.get("contains", "")).strip().lower()
    contains_all = [str(t).lower() for t in spec.get("contains_all", []) if str(t).strip()]
    contains_any = [str(t).lower() for t in spec.get("contains_any", []) if str(t).strip()]

    if contains and contains not in text_lower:
        return False
    if contains_all and not all(token in text_lower for token in contains_all):
        return False
    if contains_any and not any(token in text_lower for token in contains_any):
        return False
    return bool(contains or contains_all or contains_any)


def resolve_positive_filters(
    filters: List[Dict[str, Any]],
    chunk_map: Dict[str, Dict[str, Any]],
    *,
    return_details: bool = False,
) -> Any:
    if not filters:
        return ([], []) if return_details else []
    resolved: List[str] = []
    details: List[Dict[str, Any]] = []
    for filter_index, filter_spec in enumerate(filters):
        fields = filter_spec.get("fields") or [filter_spec.get("field") or "chunk_summary"]
        limit = int(filter_spec.get("max_matches", 5))
        matches_for_filter = 0
        matched_ids: List[str] = []
        for chunk_id, chunk in chunk_map.items():
            for field in fields or DEFAULT_FILTER_FIELDS:
                text = get_chunk_field_value(chunk, field or "")
                if not text:
                    continue
                if _matches_filter_spec(text, filter_spec):
                    chunk_key = str(chunk_id)
                    resolved.append(chunk_key)
                    matched_ids.append(chunk_key)
                    matches_for_filter += 1
                    break
            if matches_for_filter >= limit:
                break
        details.append(
            {
                "index": filter_index,
                "fields": fields or DEFAULT_FILTER_FIELDS,
                "matched_ids": matched_ids,
                "spec": {k: v for k, v in filter_spec.items() if k != "fields"},
            }
        )
    deduped = _dedupe_preserve_order(resolved)
    if return_details:
        return deduped, details
    return deduped


@dataclass
class QueryExample:
    query_id: str
    text: str
    positives: List[str]
    difficulty: str
    split: str
    metadata: Dict[str, Any]
    positive_filters: List[Dict[str, Any]] = field(default_factory=list)
    min_positives: int = 1
    min_positive_hits: int = 1
    positive_sources: Dict[str, Any] = field(default_factory=dict)


def collect_ground_truth(
    raw_query: Dict[str, Any],
    chunk_map: Dict[str, Dict[str, Any]],
    default_min_positives: int,
    *,
    fail_on_missing_ground_truth: bool,
    logger: logging.Logger,
) -> Tuple[List[str], Dict[str, Any], int]:
    raw_pos = raw_query.get("positives") or raw_query.get("positive_ids") or []
    explicit_ids = [str(item.get("id") if isinstance(item, dict) else item) for item in raw_pos]
    explicit_ids = [pid for pid in explicit_ids if pid]
    filters = raw_query.get("positive_filters") or []
    filter_matches, filter_details = resolve_positive_filters(filters, chunk_map, return_details=True)
    positives = _dedupe_preserve_order(explicit_ids + filter_matches)
    min_positives = int(raw_query.get("min_positives") or raw_query.get("min_results") or default_min_positives or 1)
    sources: Dict[str, Any] = {}
    if explicit_ids:
        sources["explicit_ids"] = explicit_ids
    if filter_details:
        sources["filters"] = filter_details
    if len(positives) >= min_positives:
        return positives, sources, min_positives

    query_id = raw_query.get("id") or raw_query.get("query_id") or raw_query.get("query")
    message = (
        f"Query {query_id} resolved {len(positives)} positives but requires at least {min_positives}. "
        "Provide positive_ids or broaden positive_filters to match production."
    )
    if fail_on_missing_ground_truth:
        raise SystemExit(message)
    logger.warning(message)
    return [], sources, min_positives


def load_queries(
    path: Path,
    chunk_map: Dict[str, Dict[str, Any]],
    *,
    default_min_positives: int,
    default_min_positive_hits: int,
    fail_on_missing_ground_truth: bool,
    logger: logging.Logger,
) -> Tuple[List[QueryExample], int]:
    if not path.exists():
        raise FileNotFoundError(f"Labeled queries file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("queries", payload) if isinstance(payload, dict) else payload
    queries: List[QueryExample] = []
    skipped = 0
    for idx, raw in enumerate(records):
        query_text = (raw.get("query") or raw.get("text") or "").strip()
        if not query_text:
            continue
        positives, sources, min_positives = collect_ground_truth(
            raw,
            chunk_map,
            default_min_positives,
            fail_on_missing_ground_truth=fail_on_missing_ground_truth,
            logger=logger,
        )
        if not positives:
            skipped += 1
            continue
        min_hits = int(raw.get("min_positive_hits") or default_min_positive_hits or 1)
        query = QueryExample(
            query_id=str(raw.get("id") or raw.get("query_id") or f"q{idx+1:04d}"),
            text=query_text,
            positives=[pid.strip() for pid in positives],
            difficulty=str(raw.get("difficulty", "hard")).lower(),
            split=str(raw.get("split", "train")).lower(),
            metadata={
                k: v
                for k, v in raw.items()
                if k
                not in {
                    "query",
                    "text",
                    "positives",
                    "positive_ids",
                    "positive_filters",
                    "min_positives",
                    "min_results",
                    "min_positive_hits",
                }
            },
            positive_filters=raw.get("positive_filters") or [],
            min_positives=max(1, min_positives),
            min_positive_hits=max(1, min_hits),
            positive_sources=sources,
        )
        queries.append(query)
    if not queries:
        raise ValueError(f"No labeled queries could be parsed from {path}")
    logger.info("Loaded %s labeled queries from %s", len(queries), path)
    return queries, skipped


__all__ = [
    "QueryExample",
    "collect_ground_truth",
    "load_queries",
    "resolve_positive_filters",
]
