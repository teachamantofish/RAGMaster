"""Hybrid reranker evaluation.

This script mirrors the embed evaluator's friendly output while adding ranking-first metrics
that make sense for a two-stage (retriever + reranker) system. It compares baseline bi-encoder
scores against the cross-encoder reranker so we can quantify the uplift per difficulty split.
"""

import argparse
import json
import math
from argparse import BooleanOptionalAction
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

import chunks_loader
from build_query_definition import load_queries
from create_chunk_candidate_list import (
    BM25Lite,
    build_bm25_index,
    encode_corpus,
    score_query_candidates,
    validate_negative_hits,
    validate_positive_hits,
)
from export_pairs import format_query_table, format_validation_summary

from scripts.config_training_rerank import (
    CROSS_ENCODER_DATA_DIR,
    RERANKER_OUTPUT_PATH,
    RETRIEVER_BASELINE_MODEL_PATH,
    HYBRID_PIPELINE_CONFIG,
    RERANK_EVAL_CONFIG,
    RERANKER_TRAINING_CONFIG,
    CONFIG_MODEL_NAME,
    RETRIEVER_PIPELINE_CONFIG,
)
from scripts.custom_logger import setup_global_logger


LOG_HEADER = ["Date", "Level", "Message", "Difficulty", "Metric"]
logger = setup_global_logger(
    script_name="7evaluate_model",
    cwd=CROSS_ENCODER_DATA_DIR,
    log_level="INFO",
    headers=LOG_HEADER,
)


def _resolve_reranker_model_id(model_candidate: str) -> str:
    """Return a valid CrossEncoder checkpoint, falling back to the base model if needed."""

    candidate_path = Path(model_candidate)
    if candidate_path.exists():
        config_present = (candidate_path.is_file()) or (candidate_path / "config.json").exists()
        if config_present:
            return str(candidate_path)

        logger.warning(
            "No CrossEncoder checkpoint artifacts found at %s; falling back to base model %s",
            candidate_path,
            CONFIG_MODEL_NAME,
        )
        return CONFIG_MODEL_NAME

    return model_candidate


def _load_pairs(difficulty: str, split: str) -> List[Dict]:
    path = CROSS_ENCODER_DATA_DIR / difficulty / f"{split}.jsonl"
    if not path.exists():
        logger.warning("Pair file missing: %s", path)
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _batchify(iterable: List, batch_size: int) -> Iterable[List]:
    for idx in range(0, len(iterable), batch_size):
        yield iterable[idx : idx + batch_size]


def _encode_baseline(model: SentenceTransformer, texts: List[str], batch_size: int) -> Dict[str, torch.Tensor]:
    unique_texts = list(dict.fromkeys(texts))
    embeddings = model.encode(
        unique_texts,
        convert_to_tensor=True,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return {text: emb for text, emb in zip(unique_texts, embeddings)}


def _score_baseline(model: SentenceTransformer, pairs: List[Dict], batch_size: int) -> List[float]:
    anchor_cache = _encode_baseline(model, [p["query"] for p in pairs], batch_size)
    cand_cache = _encode_baseline(model, [p["candidate"] for p in pairs], batch_size)
    scores = []
    for pair in pairs:
        anchor = anchor_cache[pair["query"]]
        candidate = cand_cache[pair["candidate"]]
        sim = torch.nn.functional.cosine_similarity(anchor, candidate, dim=0)
        scores.append(float(sim))
    return scores


def _score_reranker(model: CrossEncoder, pairs: List[Dict], batch_size: int) -> List[float]:
    sentences = [[pair["query"], pair["candidate"]] for pair in pairs]
    scores = []
    for batch in _batchify(sentences, batch_size):
        batch_scores = model.predict(batch)
        scores.extend(batch_scores.tolist())
    return scores


def _group_by_anchor(pairs: List[Dict]) -> Dict[Tuple[str, str], List[Dict]]:
    buckets: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for record in pairs:
        buckets[(record["difficulty"], record["query"])] += [record]
    return buckets


def _dcg(labels: List[int], k: int) -> float:
    return sum((label / math.log2(idx + 2)) for idx, label in enumerate(labels[:k]))


def _ndcg(labels: List[int], k: int) -> float:
    ideal = sorted(labels, reverse=True)
    denom = _dcg(ideal, k)
    if denom == 0:
        return 0.0
    return _dcg(labels, k) / denom


def _anchor_metrics(anchor_groups: Dict[Tuple[str, str], List[Dict]], score_field: str, k_values: List[int]):
    totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for (difficulty, _), records in anchor_groups.items():
        positives = [r for r in records if r["label"] == 1]
        negatives = [r for r in records if r["label"] == 0]
        if not positives or not negatives:
            continue

        sorted_records = sorted(records, key=lambda r: r[score_field], reverse=True)
        labels = [rec["label"] for rec in sorted_records]

        top_label = labels[0] if labels else 0
        totals[difficulty]["top1_positive_rate"] += 1.0 if top_label == 1 else 0.0
        counts[difficulty]["top1_positive_rate"] += 1

        # Pair accuracy: did we rank a positive above the best negative?
        if max(rec[score_field] for rec in positives) > max(rec[score_field] for rec in negatives):
            totals[difficulty]["pair_accuracy"] += 1.0
        else:
            totals[difficulty]["pair_accuracy"] += 0.0
        counts[difficulty]["pair_accuracy"] += 1

        # Reciprocal rank of first relevant item
        try:
            first_rel = labels.index(1)
            totals[difficulty]["mrr"] += 1 / (first_rel + 1)
        except ValueError:
            totals[difficulty]["mrr"] += 0.0
        counts[difficulty]["mrr"] += 1

        # Per-k metrics
        relevant_total = sum(labels)
        for k in k_values:
            prefix = labels[:k]
            totals[difficulty][f"recall@{k}"] += (sum(prefix) / relevant_total) if relevant_total else 0.0
            totals[difficulty][f"ndcg@{k}"] += _ndcg(labels, k)
            counts[difficulty][f"recall@{k}"] += 1
            counts[difficulty][f"ndcg@{k}"] += 1

    metrics = {}
    for difficulty, metric_totals in totals.items():
        metrics[difficulty] = {
            name: (metric_totals[name] / counts[difficulty][name])
            for name in metric_totals
            if counts[difficulty][name]
        }
    return metrics


def _merge_scores(pairs: List[Dict], baseline_scores: List[float], reranker_scores: List[float]) -> List[Dict]:
    enriched = []
    for pair, base_score, rerank_score in zip(pairs, baseline_scores, reranker_scores):
        enriched.append({
            **pair,
            "baseline_score": base_score,
            HYBRID_PIPELINE_CONFIG["score_field"]: rerank_score,
        })
    return enriched


def _summarize(enriched_pairs: List[Dict], k_values: List[int]):
    baseline_groups = _group_by_anchor(enriched_pairs)
    rerank_groups = _group_by_anchor(enriched_pairs)

    baseline_metrics = _anchor_metrics(baseline_groups, "baseline_score", k_values)
    rerank_metrics = _anchor_metrics(
        rerank_groups,
        HYBRID_PIPELINE_CONFIG["score_field"],
        k_values,
    )

    summary = {}
    for difficulty in rerank_metrics.keys() | baseline_metrics.keys():
        summary[difficulty] = {
            "pairs": sum(1 for rec in enriched_pairs if rec["difficulty"] == difficulty),
            "anchors": len({rec["query"] for rec in enriched_pairs if rec["difficulty"] == difficulty}),
            "baseline": baseline_metrics.get(difficulty, {}),
            "reranker": rerank_metrics.get(difficulty, {}),
        }
    return summary


def _print_table(summary: Dict[str, Dict], k_values: List[int], aggregate: Dict[str, Dict] | None):
    headers = [
        "Difficulty",
        "Anchors",
        "PairAcc (base → rerank)",
        "MRR (base → rerank)",
        "Top1% (base → rerank)",
    ] + [f"Recall@{k} (base → rerank)" for k in k_values] + [f"NDCG@{k} (base → rerank)" for k in k_values]

    print("\n" + "=" * 80)
    print("📊 RERANKER VS BASELINE")
    print("=" * 80)
    print(" | ".join(f"{h:>15}" for h in headers))
    print("-" * 80)

    ordered_keys = [key for key in RERANK_EVAL_CONFIG["difficulties"] if key in summary]
    if "all" not in ordered_keys:
        ordered_keys.append("all")

    for difficulty in ordered_keys:
        if difficulty == "all":
            agg = aggregate
            if not agg:
                continue
            row = _format_row("ALL", agg, k_values)
        else:
            stats = summary.get(difficulty)
            if not stats:
                continue
            row = _format_row(difficulty.capitalize(), stats, k_values)
        print(" | ".join(f"{cell:>15}" for cell in row))

    print("=" * 80 + "\n")


def _aggregate_summary(summary: Dict[str, Dict]):
    if not summary:
        return None
    agg = {"pairs": 0, "anchors": 0, "baseline": defaultdict(float), "reranker": defaultdict(float)}
    for stats in summary.values():
        agg["pairs"] += stats.get("pairs", 0)
        agg["anchors"] += stats.get("anchors", 0)
        for key, value in stats.get("baseline", {}).items():
            agg["baseline"][key] += value
        for key, value in stats.get("reranker", {}).items():
            agg["reranker"][key] += value
    count = len(summary)
    agg["baseline"] = {k: v / count for k, v in agg["baseline"].items() if count}
    agg["reranker"] = {k: v / count for k, v in agg["reranker"].items() if count}
    return agg


def _format_row(label: str, stats: Dict, k_values: List[int]) -> List[str]:
    baseline = stats.get("baseline", {})
    reranker = stats.get("reranker", {})
    row = [
        label,
        str(stats.get("anchors", 0)),
        f"{baseline.get('pair_accuracy', 0):.2f} → {reranker.get('pair_accuracy', 0):.2f}",
        f"{baseline.get('mrr', 0):.2f} → {reranker.get('mrr', 0):.2f}",
        f"{baseline.get('top1_positive_rate', 0):.2f} → {reranker.get('top1_positive_rate', 0):.2f}",
    ]
    for k in k_values:
        row.append(
            f"{baseline.get(f'recall@{k}', 0):.2f} → {reranker.get(f'recall@{k}', 0):.2f}"
        )
    for k in k_values:
        row.append(
            f"{baseline.get(f'ndcg@{k}', 0):.2f} → {reranker.get(f'ndcg@{k}', 0):.2f}"
        )
    return row


def _write_outputs(
    enriched_pairs: List[Dict],
    summary: Dict[str, Dict],
    output_dir: Path,
    *,
    coverage: Dict[str, Any] | None = None,
    aggregate: Dict[str, Dict] | None = None,
    k_values: List[int] | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "reranker_eval_pairs.jsonl"
    with open(pairs_path, "w", encoding="utf-8") as handle:
        for record in enriched_pairs:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = output_dir / "reranker_eval_summary.json"
    payload = {
        "summary": summary,
    }
    if aggregate:
        payload["aggregate"] = aggregate
    if coverage:
        payload["coverage"] = coverage
    if k_values:
        payload["k_values"] = k_values
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    if RERANK_EVAL_CONFIG.get("emit_csv", False):
        csv_path = output_dir / "reranker_eval_summary.csv"
        with open(csv_path, "w", encoding="utf-8") as handle:
            handle.write("difficulty,metric,value\n")
            for difficulty, stats in summary.items():
                for variant, metrics in ("baseline", stats.get("baseline", {})), ("reranker", stats.get("reranker", {})):
                    for metric_name, metric_value in metrics.items():
                        handle.write(f"{difficulty},{variant}:{metric_name},{metric_value:.6f}\n")
        logger.info("Saved CSV summary to %s", csv_path)

    logger.info("Saved detailed pair scores to %s", pairs_path)
    logger.info("Saved summary metrics to %s", summary_path)


def _ensure_padding_token(cross_encoder: CrossEncoder) -> None:
    """Make sure the tokenizer and config have a valid padding token/id."""

    tokenizer = cross_encoder.tokenizer

    pad_token = tokenizer.pad_token
    pad_token_id = tokenizer.pad_token_id

    if pad_token is None or pad_token_id is None:
        fallback_token = None
        fallback_id = None
        for token_attr, id_attr in (
            ("pad_token", "pad_token_id"),
            ("eos_token", "eos_token_id"),
            ("sep_token", "sep_token_id"),
            ("cls_token", "cls_token_id"),
        ):
            token_value = getattr(tokenizer, token_attr, None)
            token_id = getattr(tokenizer, id_attr, None)
            if token_value is not None:
                fallback_token = token_value
                fallback_id = token_id or tokenizer.convert_tokens_to_ids(token_value)
                break

        if fallback_token is None or fallback_id is None:
            raise SystemExit(
                "CrossEncoder tokenizer is missing a pad/eos/sep token; please update the checkpoint to define one."
            )

        tokenizer.pad_token = fallback_token
        tokenizer.pad_token_id = fallback_id
        pad_token = fallback_token
        pad_token_id = fallback_id
        logger.info("Tokenizer lacked pad token; reusing %s as padding", fallback_token)

    if cross_encoder.model.config.pad_token_id is None:
        cross_encoder.model.config.pad_token_id = pad_token_id


def _build_live_retrieval_pairs(
    args,
    baseline_model: SentenceTransformer,
):
    """Run the hybrid retriever to materialize per-query candidate lists."""

    chunk_list, chunk_map = chunks_loader.load_chunks(RETRIEVER_PIPELINE_CONFIG, logger=logger)
    chunk_ids = [chunk["id"] for chunk in chunk_list]
    chunk_texts = [chunk["text"] for chunk in chunk_list]
    bm25 = build_bm25_index(chunk_list)
    chunk_embeddings = encode_corpus(baseline_model, chunk_list, args.retrieval_batch_size)

    queries, skipped_ground_truth = load_queries(
        Path(args.queries_file),
        chunk_map,
        default_min_positives=args.min_ground_truth,
        default_min_positive_hits=args.min_positive_hits,
        fail_on_missing_ground_truth=not args.allow_missing_ground_truth,
        logger=logger,
    )

    allowed_difficulties = {difficulty.lower() for difficulty in args.difficulties}
    allowed_splits = {args.split} if args.split != "all" else {"train", "test"}
    filtered_queries = [
        query
        for query in queries
        if query.split in allowed_splits and (not allowed_difficulties or query.difficulty in allowed_difficulties)
    ]
    if args.retrieval_max_queries:
        filtered_queries = filtered_queries[: args.retrieval_max_queries]

    if not filtered_queries:
        raise SystemExit("Live retrieval produced no queries to evaluate. Check --split/--difficulties filters.")

    pairs: List[Dict] = []
    query_breakdown: List[Dict[str, Any]] = []
    validation_issues = []

    for query in filtered_queries:
        candidates = score_query_candidates(
            query,
            baseline_model,
            chunk_embeddings,
            bm25,
            chunk_ids,
            chunk_texts,
            args.retrieval_top_k,
            args.retrieval_lexical_weight,
            args.retrieval_dense_weight,
        )
        positives_in_topk = sum(cand.label for cand in candidates)
        negatives_in_topk = max(0, len(candidates) - positives_in_topk)
        if positives_in_topk == 0:
            logger.warning(
                "Query %s has no positives in top-%s",
                query.query_id,
                args.retrieval_top_k,
                extra={"Query": query.text, "Stage": "retrieval"},
            )
        issue = validate_positive_hits(query, positives_in_topk)
        if issue:
            validation_issues.append(issue)
            details = issue.details
            logger.error(
                "Query %s failed coverage (required=%s found=%s)",
                query.query_id,
                details.get("required"),
                details.get("found"),
                extra={"Query": query.text, "Stage": "validation"},
            )
        neg_issue = validate_negative_hits(query, negatives_in_topk)
        if neg_issue:
            validation_issues.append(neg_issue)
            logger.error(
                "Query %s lacks negative coverage in top-%s",
                query.query_id,
                args.retrieval_top_k,
                extra={"Query": query.text, "Stage": "validation"},
            )

        for cand in candidates:
            pairs.append(
                {
                    "query_id": query.query_id,
                    "query": query.text,
                    "candidate_id": cand.candidate_id,
                    "candidate": cand.text,
                    "label": cand.label,
                    "difficulty": query.difficulty,
                    "split": query.split,
                    "rank": cand.rank,
                    "scores": {
                        "dense": cand.score_dense,
                        "lexical": cand.score_lexical,
                        "combined": cand.score_combined,
                    },
                }
            )
        top_rank = min((cand.rank for cand in candidates if cand.label == 1), default=None)
        query_breakdown.append(
            {
                "query_id": query.query_id,
                "difficulty": query.difficulty,
                "split": query.split,
                "positives_total": len(query.positives),
                "positives_in_topk": positives_in_topk,
                "negatives_in_topk": negatives_in_topk,
                "top_positive_rank": top_rank,
                "pairs": len(candidates),
            }
        )

    coverage = {
        "chunk_count": len(chunk_list),
        "queries_total": len(filtered_queries),
        "queries_skipped_ground_truth": skipped_ground_truth,
        "queries_failed_topk": len(validation_issues),
        "query_breakdown": query_breakdown,
    }

    return pairs, coverage, validation_issues


def main():
    parser = argparse.ArgumentParser(description="Evaluate reranker vs embed baseline")
    parser.add_argument(
        "--difficulties",
        nargs="*",
        default=RERANK_EVAL_CONFIG["difficulties"],
        help="Difficulty splits to evaluate (default: all)",
    )
    parser.add_argument(
        "--split",
        default=RERANK_EVAL_CONFIG.get("default_split", "test"),
        choices=["train", "test", "all"],
        help="Which pair split to evaluate (train/test/all)",
    )
    parser.add_argument(
        "--model-path",
        default=str(RERANKER_OUTPUT_PATH),
        help="Cross-encoder checkpoint to evaluate",
    )
    parser.add_argument(
        "--baseline-model",
        default=str(RETRIEVER_BASELINE_MODEL_PATH),
        help="Bi-encoder baseline model path",
    )
    parser.add_argument(
        "--baseline-batch-size",
        type=int,
        default=32,
        help="Batch size for SentenceTransformer encoding",
    )
    parser.add_argument(
        "--cross-batch-size",
        type=int,
        default=HYBRID_PIPELINE_CONFIG.get("reranker_batch_size", 64),
        help="Batch size for CrossEncoder scoring",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CROSS_ENCODER_DATA_DIR / "eval_outputs",
        help="Directory to store JSON/JSONL outputs",
    )
    parser.add_argument(
        "--use-live-retrieval",
        action=BooleanOptionalAction,
        default=RERANK_EVAL_CONFIG.get("use_live_retrieval", True),
        help="When true, rebuild candidate lists via the hybrid retriever before scoring",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=RETRIEVER_PIPELINE_CONFIG.get("queries_file"),
        help="Labeled query file used during live retrieval",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=RETRIEVER_PIPELINE_CONFIG.get("top_k", 50),
        help="Number of candidates per query for live retrieval",
    )
    parser.add_argument(
        "--retrieval-lexical-weight",
        type=float,
        default=RETRIEVER_PIPELINE_CONFIG.get("lexical_weight", 0.35),
        help="BM25 weight inside the hybrid score",
    )
    parser.add_argument(
        "--retrieval-dense-weight",
        type=float,
        default=RETRIEVER_PIPELINE_CONFIG.get("dense_weight", 0.65),
        help="Dense encoder weight inside the hybrid score",
    )
    parser.add_argument(
        "--retrieval-batch-size",
        type=int,
        default=32,
        help="Batch size used to encode chunks for live retrieval",
    )
    parser.add_argument(
        "--retrieval-max-queries",
        type=int,
        default=RETRIEVER_PIPELINE_CONFIG.get("max_queries"),
        help="Optional cap on labeled queries during live retrieval",
    )
    parser.add_argument(
        "--min-ground-truth",
        type=int,
        default=RETRIEVER_PIPELINE_CONFIG.get("min_ground_truth", 1),
        help="Minimum positives each query definition must resolve",
    )
    parser.add_argument(
        "--min-positive-hits",
        type=int,
        default=RETRIEVER_PIPELINE_CONFIG.get("min_positive_hits", 1),
        help="Minimum positives that must appear in the retrieved top-k",
    )
    parser.add_argument(
        "--allow-missing-ground-truth",
        action=BooleanOptionalAction,
        default=RERANK_EVAL_CONFIG.get("allow_missing_ground_truth", False),
        help="Continue even if positive filters resolve fewer than the required IDs",
    )
    parser.add_argument(
        "--allow-missed-positives",
        action=BooleanOptionalAction,
        default=RERANK_EVAL_CONFIG.get("allow_missed_positives", False),
        help="Continue even when positives are missing from the retrieved top-k",
    )

    args = parser.parse_args()
    args.difficulties = [d.lower() for d in args.difficulties]

    baseline_model = SentenceTransformer(args.baseline_model)
    selected_pairs: List[Dict] = []
    coverage = None
    validation_issues = []

    if args.use_live_retrieval:
        selected_pairs, coverage, validation_issues = _build_live_retrieval_pairs(args, baseline_model)
    else:
        splits = [args.split] if args.split != "all" else ["train", "test"]
        for split in splits:
            for difficulty in args.difficulties:
                pairs = _load_pairs(difficulty, split)
                if not pairs:
                    logger.warning("No %s pairs found for %s split", difficulty, split)
                    continue
                selected_pairs.extend(pairs)

    if not selected_pairs:
        raise SystemExit(
            "No evaluation data found. Run build_retrieval_candidates.py or enable --use-live-retrieval."
        )

    """
    3evaluate_model.py with --use-live-retrieval (default), it rebuilds the candidate lists 
    on the fly via _build_live_retrieval_pairs, then reuses _format_query_table from 
    build_retrieval_candidates.py to emit this coverage block before scoring the reranker. 
    It sanity-checks the retrieval stage that feeds the reranker by showing, per labeled query, 
    how many of its ground-truth positives were actually retrieved inside the hybrid top‑K slate. 
    If these numbers look bad, the reranker evaluation can’t prove anything because the relevant 
    documents never reach stage two.
    """
    if coverage and coverage.get("query_breakdown"):
        table = format_query_table(coverage["query_breakdown"])
        if table:
            print("\n" + "=" * 80)
            print("📊 evaluate_model.py: Retrieval Coverage evaluation")
            print("=" * 80)
            print(table)
            print("=" * 80 + "\n")
            logger.debug("evaluate_model.py: Retrieval coverage evaluation:\n%s", table)

    if validation_issues and not args.allow_missed_positives:
        summary = format_validation_summary(validation_issues)
        raise SystemExit(summary)
    elif validation_issues:
        summary = format_validation_summary(validation_issues)
        logger.warning("Continuing despite validation issues:\n%s", summary)

    logger.info("Loaded %d pairs across %d difficulties", len(selected_pairs), len(args.difficulties))

    reranker_model_id = _resolve_reranker_model_id(args.model_path)
    reranker_model = CrossEncoder(
        reranker_model_id,
        max_length=RERANKER_TRAINING_CONFIG.get("max_length", 512),
    )
    _ensure_padding_token(reranker_model)

    if args.use_live_retrieval or all(
        (record.get("scores") and "combined" in record["scores"]) for record in selected_pairs
    ):
        baseline_scores = [record.get("scores", {}).get("combined", 0.0) for record in selected_pairs]
    else:
        baseline_scores = _score_baseline(baseline_model, selected_pairs, args.baseline_batch_size)
    reranker_scores = _score_reranker(reranker_model, selected_pairs, args.cross_batch_size)
    enriched_pairs = _merge_scores(selected_pairs, baseline_scores, reranker_scores)

    k_values = sorted({5, 10, HYBRID_PIPELINE_CONFIG.get("reranker_top_k", 20)})
    summary = _summarize(enriched_pairs, k_values)
    aggregate = _aggregate_summary(summary)
    _print_table(summary, k_values, aggregate)

    if RERANK_EVAL_CONFIG.get("emit_json", True):
        _write_outputs(
            enriched_pairs,
            summary,
            args.output_dir,
            coverage=coverage,
            aggregate=aggregate,
            k_values=k_values,
        )


if __name__ == "__main__":
    main()
