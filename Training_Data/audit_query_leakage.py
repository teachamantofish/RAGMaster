from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", (text or ""))


def _api_like_tokens(tokens: List[str]) -> List[str]:
    api = []
    for tok in tokens:
        if "_" in tok or re.search(r"[A-Z]", tok) or any(ch.isdigit() for ch in tok):
            api.append(tok.lower())
    return api


def _norm_set(items: List[str]) -> set[str]:
    return {item.strip().lower() for item in items if item and item.strip()}


def _build_chunk_map(chunks_file: Path) -> Dict[str, dict]:
    payload = json.loads(chunks_file.read_text(encoding="utf-8"))
    chunks = payload.get("chunks") if isinstance(payload, dict) and "chunks" in payload else payload
    if not isinstance(chunks, list):
        raise ValueError(f"Expected list in {chunks_file}")
    out = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "").strip()
        if chunk_id:
            out[chunk_id] = chunk
    return out


def _chunk_text(chunk: dict) -> str:
    return "\n".join(
        [
            str(chunk.get("heading") or ""),
            str(chunk.get("concat_header_path") or ""),
            str(chunk.get("chunk_summary") or ""),
            str(chunk.get("content") or ""),
            str(chunk.get("code_friendly_name") or ""),
        ]
    )


def _analyze_query(query: dict, chunk_map: Dict[str, dict], high_overlap_threshold: float) -> dict:
    query_text = str(query.get("query") or "")
    query_tokens = _tokenize(query_text)
    query_set = _norm_set(query_tokens)

    pos_ids = [str(pid) for pid in query.get("positive_ids") or []]
    pos_chunks = [chunk_map[pid] for pid in pos_ids if pid in chunk_map]
    pos_text = "\n".join(_chunk_text(chunk) for chunk in pos_chunks)
    pos_tokens = _tokenize(pos_text)
    pos_set = _norm_set(pos_tokens)

    overlap = query_set & pos_set
    token_overlap_ratio = (len(overlap) / len(query_set)) if query_set else 0.0

    query_api = _norm_set(_api_like_tokens(query_tokens))
    pos_api = _norm_set(_api_like_tokens(pos_tokens))
    api_overlap = query_api & pos_api
    api_overlap_ratio = (len(api_overlap) / len(query_api)) if query_api else 0.0

    heading_hits = 0
    q_lower = query_text.lower()
    for chunk in pos_chunks:
        heading = str(chunk.get("heading") or "").strip().lower()
        if heading and heading in q_lower:
            heading_hits += 1

    return {
        "id": query.get("id"),
        "source": query.get("source", "unknown"),
        "token_overlap_ratio": round(token_overlap_ratio, 4),
        "api_overlap_ratio": round(api_overlap_ratio, 4),
        "heading_substring_hits": heading_hits,
        "is_high_overlap": token_overlap_ratio >= high_overlap_threshold,
        "query_token_count": len(query_set),
        "positive_count": len(pos_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit query leakage against positive chunks")
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=Path("Training_Data/retriever_eval_queries.json"),
        help="Query definitions JSON file",
    )
    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=Path("Data/framemaker/mif_jsx/a_chunks.json"),
        help="Chunk corpus JSON",
    )
    parser.add_argument(
        "--high-overlap-threshold",
        type=float,
        default=0.60,
        help="Threshold for high query-token overlap with positives",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Training_Data/leakage_audit_report.json"),
        help="Output JSON report path",
    )
    args = parser.parse_args()

    queries = json.loads(args.queries_file.read_text(encoding="utf-8"))
    if not isinstance(queries, list):
        raise ValueError(f"Expected list in {args.queries_file}")

    chunk_map = _build_chunk_map(args.chunks_file)
    rows = [_analyze_query(query, chunk_map, args.high_overlap_threshold) for query in queries]

    by_source = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source", "unknown"))].append(row)

    def _avg(values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    summary = {
        "queries_file": str(args.queries_file),
        "chunks_file": str(args.chunks_file),
        "query_count": len(rows),
        "high_overlap_threshold": args.high_overlap_threshold,
        "high_overlap_count": sum(1 for row in rows if row["is_high_overlap"]),
        "high_overlap_pct": round(100.0 * sum(1 for row in rows if row["is_high_overlap"]) / max(1, len(rows)), 2),
        "avg_token_overlap_ratio": _avg([row["token_overlap_ratio"] for row in rows]),
        "avg_api_overlap_ratio": _avg([row["api_overlap_ratio"] for row in rows]),
        "source_breakdown": {},
    }

    for source, source_rows in by_source.items():
        summary["source_breakdown"][source] = {
            "count": len(source_rows),
            "high_overlap_pct": round(
                100.0 * sum(1 for row in source_rows if row["is_high_overlap"]) / max(1, len(source_rows)),
                2,
            ),
            "avg_token_overlap_ratio": _avg([row["token_overlap_ratio"] for row in source_rows]),
            "avg_api_overlap_ratio": _avg([row["api_overlap_ratio"] for row in source_rows]),
        }

    top_high = sorted(rows, key=lambda row: row["token_overlap_ratio"], reverse=True)[:25]
    report = {"summary": summary, "top_high_overlap_examples": top_high, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[AUDIT] Wrote report: {args.output}")
    print(f"[AUDIT] Queries: {summary['query_count']}")
    print(f"[AUDIT] High-overlap: {summary['high_overlap_count']} ({summary['high_overlap_pct']}%)")
    print(f"[AUDIT] Avg token overlap: {summary['avg_token_overlap_ratio']}")
    print(f"[AUDIT] Avg API overlap: {summary['avg_api_overlap_ratio']}")


if __name__ == "__main__":
    main()
