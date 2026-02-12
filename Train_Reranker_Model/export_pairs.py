from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.config_training_rerank import (
    BASE_CWD,
    CROSS_ENCODER_DATA_DIR,
    RETRIEVER_PIPELINE_CONFIG,
    TRAINING_DATA_DIR,
)

LOG_TARGET = BASE_CWD / RETRIEVER_PIPELINE_CONFIG.get("log_filename", "build_retrieval_candidates.log")


def export_pairs(pairs: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> None:
    for difficulty, splits in pairs.items():
        diff_dir = CROSS_ENCODER_DATA_DIR / difficulty
        diff_dir.mkdir(parents=True, exist_ok=True)
        for split, records in splits.items():
            output_path = diff_dir / f"{split}.jsonl"
            with open(output_path, "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_triplets(triplets: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> None:
    for difficulty, splits in triplets.items():
        diff_dir = TRAINING_DATA_DIR / difficulty
        diff_dir.mkdir(parents=True, exist_ok=True)
        for split, records in splits.items():
            output_path = diff_dir / f"triplets_{split}.json"
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(records, handle, indent=2, ensure_ascii=False)


def format_query_table(query_stats: List[Dict[str, Any]]) -> str:
    if not query_stats:
        return ""
    header = f"{'Query':<8} {'Split':<6} {'Diff':<6} {'Pos':>5} {'Pos@K':>6} {'Neg@K':>6} {'Recall':>7} {'Top':>5} {'Pairs':>6}"
    lines = [header, "-" * len(header)]
    for stat in query_stats:
        recall = (
            stat["positives_in_topk"] / stat["positives_total"]
            if stat["positives_total"]
            else 0.0
        )
        top_rank = stat.get("top_positive_rank")
        lines.append(
            f"{stat['query_id']:<8} {stat['split']:<6} {stat['difficulty']:<6} "
            f"{stat['positives_total']:>5} {stat['positives_in_topk']:>6} {stat.get('negatives_in_topk', 0):>6} {recall:>7.2f} "
            f"{(top_rank if top_rank is not None else '—'):>5} {stat['pairs']:>6}"
        )
    return "\n".join(lines)


def write_summary_log(
    stats: Dict[str, Any],
    query_stats: List[Dict[str, Any]],
    issues: Optional[List[Any]] = None,
    *,
    log_target: Path = LOG_TARGET,
) -> None:
    log_target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(stats)
    if query_stats:
        payload["query_breakdown"] = query_stats
    if issues:
        payload["validation_issues"] = [getattr(issue, "__dict__", issue) for issue in issues]
    with open(log_target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def format_validation_summary(issues: List[Any]) -> str:
    lines = ["❌ Query coverage issues detected:"]
    for issue in issues:
        details = getattr(issue, "details", {})
        lines.append(
            f"  - {getattr(issue, 'query_id', 'unknown')}: {getattr(issue, 'reason', 'unknown')} "
            f"(required={details.get('required')}, found={details.get('found')})"
        )
    return "\n".join(lines)


__all__ = [
    "export_pairs",
    "export_triplets",
    "format_query_table",
    "format_validation_summary",
    "write_summary_log",
]
