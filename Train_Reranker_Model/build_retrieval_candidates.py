"""Build retrieval-driven candidate sets for reranker training/evaluation."""

from __future__ import annotations

import argparse
import json
from argparse import BooleanOptionalAction
from pathlib import Path
from typing import Any, Dict, List

import torch
from sentence_transformers import SentenceTransformer

import chunks_loader
from build_query_definition import load_queries
from create_chunk_candidate_list import (
    build_bm25_index,
    build_candidate_pairs,
    encode_corpus,
    probe_test_question,
)
from export_pairs import (
    export_pairs,
    export_triplets,
    format_query_table,
    format_validation_summary,
    write_summary_log,
)
from scripts.config_training_rerank import (
    BASE_CWD,
    RETRIEVER_BASELINE_MODEL_PATH,
    RETRIEVER_PIPELINE_CONFIG,
)
from scripts.custom_logger import setup_global_logger

SCRIPT_NAME = "build_retrieval_candidates"
LOG_HEADERS = ["Date", "Level", "Message", "Query", "Stage"]

logger = setup_global_logger(
    script_name=SCRIPT_NAME,
    cwd=BASE_CWD,
    log_level="INFO",
    headers=LOG_HEADERS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build retrieval-driven candidate sets")
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=RETRIEVER_PIPELINE_CONFIG.get("queries_file"),
        help="JSON or JSONL file with labeled queries",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RETRIEVER_PIPELINE_CONFIG.get("top_k", 50),
        help="Number of candidates to keep per query",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=RETRIEVER_PIPELINE_CONFIG.get("max_queries"),
        help="Optional cap for smoke tests",
    )
    parser.add_argument(
        "--lexical-weight",
        type=float,
        default=RETRIEVER_PIPELINE_CONFIG.get("lexical_weight", 0.35),
        help="Weight assigned to BM25 scores when combining",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=RETRIEVER_PIPELINE_CONFIG.get("dense_weight", 0.65),
        help="Weight assigned to dense scores when combining",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--test-question",
        type=str,
        default=RETRIEVER_PIPELINE_CONFIG.get("test_question", ""),
        help="Optional probe question to sanity-check retrieval",
    )
    parser.add_argument(
        "--min-positive-in-topk",
        type=int,
        default=RETRIEVER_PIPELINE_CONFIG.get("min_positive_hits", 1),
        help="Minimum number of positives that must appear in the retrieved top-k",
    )
    parser.add_argument(
        "--min-ground-truth",
        type=int,
        default=RETRIEVER_PIPELINE_CONFIG.get("min_ground_truth", 1),
        help="Minimum number of positives each query definition must resolve before retrieval",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run retrieval + validation without exporting triplets/pairs",
    )
    parser.add_argument(
        "--fail-on-missed-positives",
        action=BooleanOptionalAction,
        default=True,
        help="Fail the run when a query lacks the required positives in top-k",
    )
    parser.add_argument(
        "--fail-on-missing-ground-truth",
        action=BooleanOptionalAction,
        default=True,
        help="Fail the run when positive filters/ids resolve fewer than the required ground-truth positives",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        help="Optional JSON file to store retrieval stats for orchestration scripts",
    )
    return parser.parse_args()


def _resolve_queries(args: argparse.Namespace) -> Path:
    configured = RETRIEVER_PIPELINE_CONFIG.get("queries_file")
    return Path(args.queries_file or configured)


def _load_chunk_corpus() -> Any:
    return chunks_loader.load_chunks(RETRIEVER_PIPELINE_CONFIG, logger=logger)


def _compute_pair_stats(pairs: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> int:
    return sum(len(records) for splits in pairs.values() for records in splits.values())


def _compute_triplet_stats(triplets: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> int:
    return sum(len(records) for splits in triplets.values() for records in splits.values())


def main() -> None:
    args = parse_args()

    chunk_list, chunk_map = _load_chunk_corpus()
    queries_file = _resolve_queries(args)
    queries, skipped_ground_truth = load_queries(
        queries_file,
        chunk_map,
        default_min_positives=args.min_ground_truth,
        default_min_positive_hits=args.min_positive_in_topk,
        fail_on_missing_ground_truth=args.fail_on_missing_ground_truth,
        logger=logger,
    )
    total_defined = len(queries) + skipped_ground_truth
    available_queries = len(queries)
    if args.max_queries:
        queries = queries[: args.max_queries]
    truncated = max(0, available_queries - len(queries))

    chunk_ids = [chunk["id"] for chunk in chunk_list]
    chunk_texts = [chunk["text"] for chunk in chunk_list]
    bm25 = build_bm25_index(chunk_list)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading retriever baseline from %s on %s", RETRIEVER_BASELINE_MODEL_PATH, device)
    model = SentenceTransformer(str(RETRIEVER_BASELINE_MODEL_PATH), device=device)
    chunk_embeddings = encode_corpus(model, chunk_list, args.batch_size)

    probe_test_question(
        args.test_question,
        model,
        chunk_embeddings,
        bm25,
        chunk_ids,
        chunk_texts,
        args.top_k,
        args.lexical_weight,
        args.dense_weight,
        logger=logger,
    )

    pairs, triplets, query_breakdown, validation_issues = build_candidate_pairs(
        queries,
        chunk_map=chunk_map,
        model=model,
        chunk_embeddings=chunk_embeddings,
        bm25=bm25,
        chunk_ids=chunk_ids,
        chunk_texts=chunk_texts,
        top_k=args.top_k,
        lexical_weight=args.lexical_weight,
        dense_weight=args.dense_weight,
        validate_only=args.validate_only,
        logger=logger,
    )

    pairs_written = _compute_pair_stats(pairs)
    triplets_written = _compute_triplet_stats(triplets)

    stats = {
        "queries_total": len(queries),
        "queries_defined": total_defined,
        "queries_truncated": truncated,
        "queries_with_positive": sum(1 for stat in query_breakdown if stat["positives_in_topk"]),
        "queries_with_negative": sum(1 for stat in query_breakdown if stat.get("negatives_in_topk")),
        "queries_failed_topk": len(validation_issues),
        "queries_skipped_ground_truth": skipped_ground_truth,
        "pairs_written": pairs_written,
        "triplets_written": triplets_written,
        "chunk_count": len(chunk_list),
    }

    stats_payload = {
        **stats,
        "top_k": args.top_k,
        "lexical_weight": args.lexical_weight,
        "dense_weight": args.dense_weight,
        "batch_size": args.batch_size,
        "queries_file": str(queries_file),
        "min_positive_in_topk": args.min_positive_in_topk,
        "min_ground_truth": args.min_ground_truth,
        "require_negative": True,
    }

    if args.validate_only:
        logger.info("Validation-only mode enabled; skipping export of triplets/pairs")
    else:
        export_triplets(triplets)
        export_pairs(pairs)

    stats.update(
        {
            "difficulties": {
                difficulty: {split: len(records) for split, records in splits.items()}
                for difficulty, splits in pairs.items()
            }
        }
    )

    table_text = format_query_table(query_breakdown)
    if table_text:
        print("\n" + "=" * 80)
        print("📊 Retrieval Coverage")
        print("=" * 80)
        print(table_text)
        print("=" * 80 + "\n")
        logger.debug("Retrieval coverage summary:\n%s", table_text, extra={"Stage": "summary"})

    write_summary_log(stats, query_breakdown, validation_issues)

    if args.stats_output:
        try:
            args.stats_output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.stats_output, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        **stats_payload,
                        "query_breakdown": query_breakdown,
                        "validation_issues": [getattr(issue, "__dict__", issue) for issue in validation_issues],
                    },
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
            logger.info("Wrote retrieval stats to %s", args.stats_output)
        except OSError as exc:
            logger.error("Failed to write stats to %s: %s", args.stats_output, exc)

    if validation_issues:
        summary = format_validation_summary(validation_issues)
        logger.error(summary)
        if args.fail_on_missed_positives:
            raise SystemExit(summary)

    logger.info("Retrieval candidate build complete", extra={"Stage": "complete", "Message": str(stats)})


if __name__ == "__main__":
    main()
