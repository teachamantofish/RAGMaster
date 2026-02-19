from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Violation:
    kind: str
    id_a: str
    id_b: str
    detail: str


def _normalize_query(text: str) -> str:
    lowered = (text or "").lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", _normalize_query(text)))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _load(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    return payload


def _is_holdout_contaminated(query: dict) -> bool:
    qid = str(query.get("id") or "")
    source = str(query.get("source") or "").lower()
    if qid.startswith("q_synth_"):
        return True
    if source == "synthetic":
        return True
    if query.get("source_chunk_id"):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate train/test/holdout query split hygiene")
    parser.add_argument(
        "--main-queries",
        type=Path,
        default=Path("Training_Data/retriever_eval_queries.json"),
        help="Main query JSON file",
    )
    parser.add_argument(
        "--holdout-queries",
        type=Path,
        default=Path("Training_Data/retriever_eval_queries_human_holdout.json"),
        help="Holdout query JSON file",
    )
    parser.add_argument(
        "--near-dup-threshold",
        type=float,
        default=0.92,
        help="Sequence similarity threshold for near-duplicate detection",
    )
    parser.add_argument(
        "--cross-split-jaccard-threshold",
        type=float,
        default=0.85,
        help="Token-set Jaccard warning threshold across splits",
    )
    parser.add_argument(
        "--fail-on-violations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero when hard violations are present",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Training_Data/query_split_validation_report.json"),
        help="Output report path",
    )
    args = parser.parse_args()

    main_queries = _load(args.main_queries)
    holdout_queries = _load(args.holdout_queries) if args.holdout_queries.exists() else []

    combined: List[Tuple[str, dict]] = []
    for q in main_queries:
        split = str(q.get("split") or "train").lower()
        combined.append((split, q))
    for q in holdout_queries:
        combined.append(("holdout", q))

    duplicates: List[Violation] = []
    near_duplicates: List[Violation] = []
    high_jaccard_warnings: List[Violation] = []
    holdout_contamination: List[str] = []

    by_norm: Dict[str, List[Tuple[str, dict]]] = {}
    for split, query in combined:
        norm = _normalize_query(str(query.get("query") or ""))
        by_norm.setdefault(norm, []).append((split, query))

    for norm, entries in by_norm.items():
        if len(entries) <= 1:
            continue
        ids = [str(item[1].get("id")) for item in entries]
        splits = {item[0] for item in entries}
        if len(splits) > 1:
            duplicates.append(Violation("duplicate_text", ids[0], ids[1], f"splits={sorted(splits)} text={norm}"))

    all_items = combined
    for i in range(len(all_items)):
        split_a, qa = all_items[i]
        id_a = str(qa.get("id"))
        text_a = str(qa.get("query") or "")
        norm_a = _normalize_query(text_a)
        tokens_a = _token_set(text_a)
        for j in range(i + 1, len(all_items)):
            split_b, qb = all_items[j]
            if split_a == split_b:
                continue
            id_b = str(qb.get("id"))
            text_b = str(qb.get("query") or "")
            norm_b = _normalize_query(text_b)
            sim = SequenceMatcher(None, norm_a, norm_b).ratio()
            if sim >= args.near_dup_threshold:
                near_duplicates.append(
                    Violation("near_duplicate_text", id_a, id_b, f"sim={sim:.3f} splits={split_a}/{split_b}")
                )

            jac = _jaccard(tokens_a, _token_set(text_b))
            if jac >= args.cross_split_jaccard_threshold:
                high_jaccard_warnings.append(
                    Violation("high_jaccard", id_a, id_b, f"jaccard={jac:.3f} splits={split_a}/{split_b}")
                )

    for q in holdout_queries:
        if _is_holdout_contaminated(q):
            holdout_contamination.append(str(q.get("id")))

    hard_violation_count = len(duplicates) + len(near_duplicates) + len(holdout_contamination)
    report = {
        "main_queries_file": str(args.main_queries),
        "holdout_queries_file": str(args.holdout_queries),
        "counts": {
            "main_queries": len(main_queries),
            "holdout_queries": len(holdout_queries),
            "duplicates": len(duplicates),
            "near_duplicates": len(near_duplicates),
            "holdout_contamination": len(holdout_contamination),
            "high_jaccard_warnings": len(high_jaccard_warnings),
            "hard_violations": hard_violation_count,
        },
        "duplicates": [v.__dict__ for v in duplicates],
        "near_duplicates": [v.__dict__ for v in near_duplicates],
        "holdout_contamination_ids": holdout_contamination,
        "high_jaccard_warnings": [v.__dict__ for v in high_jaccard_warnings[:200]],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[SPLIT] Wrote report: {args.output}")
    print(f"[SPLIT] Main queries: {len(main_queries)}")
    print(f"[SPLIT] Holdout queries: {len(holdout_queries)}")
    print(f"[SPLIT] Duplicates: {len(duplicates)}")
    print(f"[SPLIT] Near duplicates: {len(near_duplicates)}")
    print(f"[SPLIT] Holdout contamination IDs: {len(holdout_contamination)}")
    print(f"[SPLIT] High-jaccard warnings: {len(high_jaccard_warnings)}")

    if args.fail_on_violations and hard_violation_count > 0:
        raise SystemExit(
            f"Hard split-validation violations found: {hard_violation_count} "
            f"(duplicates={len(duplicates)}, near_duplicates={len(near_duplicates)}, "
            f"holdout_contamination={len(holdout_contamination)})"
        )


if __name__ == "__main__":
    main()
