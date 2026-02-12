from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from sentence_transformers import SentenceTransformer, util

from build_query_definition import QueryExample
from create_context_forllm import build_candidate_context, build_context_payload, build_query_context


class BM25Lite:
    """Lightweight BM25 scorer to avoid external dependencies."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_freqs: List[Dict[str, int]] = []
        self.doc_len: List[int] = []
        self.idf: Dict[str, float] = {}
        self.avgdl = 0.0
        self._build(corpus_tokens)

    def _build(self, corpus_tokens: Sequence[Sequence[str]]) -> None:
        df_counter: Dict[str, int] = {}
        total_len = 0
        for doc in corpus_tokens:
            freq: Dict[str, int] = {}
            for term in doc:
                freq[term] = freq.get(term, 0) + 1
            self.doc_freqs.append(freq)
            self.doc_len.append(len(doc))
            total_len += len(doc)
            for term in freq:
                df_counter[term] = df_counter.get(term, 0) + 1
        self.avgdl = (total_len / len(corpus_tokens)) if corpus_tokens else 0.0
        N = float(len(corpus_tokens)) or 1.0
        for term, freq in df_counter.items():
            self.idf[term] = math.log(1.0 + (N - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: Sequence[str]) -> List[float]:
        scores: List[float] = []
        for idx, freqs in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_len[idx] or 1
            for term in query_tokens:
                if term not in freqs:
                    continue
                idf = self.idf.get(term, 0.0)
                numerator = freqs[term] * (self.k1 + 1.0)
                denominator = freqs[term] + self.k1 * (1.0 - self.b + self.b * doc_len / (self.avgdl or doc_len))
                score += idf * (numerator / denominator)
            scores.append(score)
        return scores


@dataclass
class CandidateRecord:
    candidate_id: str
    text: str
    rank: int
    label: int
    score_dense: float
    score_lexical: float
    score_combined: float


@dataclass
class QueryValidationIssue:
    query_id: str
    reason: str
    details: Dict[str, Any]


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token]


def _normalize_scores(raw_scores: Sequence[float]) -> List[float]:
    if not raw_scores:
        return []
    max_v = max(raw_scores)
    min_v = min(raw_scores)
    if math.isclose(max_v, min_v):
        return [0.0 for _ in raw_scores]
    span = max_v - min_v
    return [(score - min_v) / span for score in raw_scores]


def _model_device(model: SentenceTransformer) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_corpus(model: SentenceTransformer, chunks: List[Dict[str, Any]], batch_size: int) -> torch.Tensor:
    texts = [chunk["text"] for chunk in chunks]
    device = _model_device(model)
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=device,
    )
    return embeddings.to(device)


def build_bm25_index(chunks: List[Dict[str, Any]]) -> BM25Lite:
    corpus_tokens = [_tokenize(chunk["text"]) for chunk in chunks]
    return BM25Lite(corpus_tokens)


def _score_query(
    query: QueryExample,
    model: SentenceTransformer,
    chunk_embeddings: torch.Tensor,
    bm25: BM25Lite,
    chunk_ids: Sequence[str],
    chunk_texts: Sequence[str],
    top_k: int,
    lexical_weight: float,
    dense_weight: float,
) -> List[CandidateRecord]:
    device = _model_device(model)
    query_vec = model.encode(
        query.text,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device,
    )
    dense_scores_tensor = util.cos_sim(query_vec, chunk_embeddings)[0]
    dense_scores = dense_scores_tensor.cpu().tolist()
    lexical_scores = bm25.get_scores(_tokenize(query.text))
    dense_norm = _normalize_scores(dense_scores)
    lexical_norm = _normalize_scores(lexical_scores)
    total_weight = lexical_weight + dense_weight
    if total_weight <= 0:
        dense_weight = 1.0
        lexical_weight = 0.0
        total_weight = 1.0
    combined_scores = []
    for idx in range(len(chunk_ids)):
        combined = (
            (lexical_norm[idx] * lexical_weight) + (dense_norm[idx] * dense_weight)
        ) / total_weight
        combined_scores.append(combined)
    ranked_indices = sorted(range(len(chunk_ids)), key=lambda i: combined_scores[i], reverse=True)[:top_k]
    normalized_positive_ids = {pid.lower() for pid in query.positives}
    candidates: List[CandidateRecord] = []
    for rank, idx in enumerate(ranked_indices, start=1):
        chunk_id = chunk_ids[idx]
        label = 1 if chunk_id.lower() in normalized_positive_ids else 0
        candidates.append(
            CandidateRecord(
                candidate_id=chunk_id,
                text=chunk_texts[idx],
                rank=rank,
                label=label,
                score_dense=dense_scores[idx],
                score_lexical=lexical_scores[idx],
                score_combined=combined_scores[idx],
            )
        )
    return candidates


def _build_triplet(query: QueryExample, candidates: List[CandidateRecord]) -> Optional[Dict[str, Any]]:
    positives = [cand for cand in candidates if cand.label == 1]
    negatives = [cand for cand in candidates if cand.label == 0]
    if not positives or not negatives:
        return None
    best_pos = positives[0]
    best_neg = negatives[0]
    return {
        "anchor": query.text,
        "positive": best_pos.text,
        "negative": best_neg.text,
        "positive_id": best_pos.candidate_id,
        "negative_id": best_neg.candidate_id,
        "difficulty": query.difficulty,
        "pair_type": "retrieval_topk",
        "query_id": query.query_id,
        "retriever_scores": {
            "positive": best_pos.score_combined,
            "negative": best_neg.score_combined,
        },
    }


def _validate_positive_hits(query: QueryExample, positives_in_topk: int) -> Optional[QueryValidationIssue]:
    required = max(0, query.min_positive_hits)
    if positives_in_topk >= required:
        return None
    return QueryValidationIssue(
        query_id=query.query_id,
        reason="positives_missing",
        details={
            "required": required,
            "found": positives_in_topk,
            "difficulty": query.difficulty,
            "split": query.split,
            "positive_ids": query.positives,
        },
    )


def _validate_negative_hits(query: QueryExample, negatives_in_topk: int) -> Optional[QueryValidationIssue]:
    if negatives_in_topk > 0:
        return None
    return QueryValidationIssue(
        query_id=query.query_id,
        reason="negatives_missing",
        details={
            "required": 1,
            "found": negatives_in_topk,
            "difficulty": query.difficulty,
            "split": query.split,
        },
    )


def tokenize_text(text: str) -> List[str]:
    """Expose the internal tokenizer for downstream evaluators."""

    return _tokenize(text)


def score_query_candidates(
    query: QueryExample,
    model: SentenceTransformer,
    chunk_embeddings: torch.Tensor,
    bm25: BM25Lite,
    chunk_ids: Sequence[str],
    chunk_texts: Sequence[str],
    top_k: int,
    lexical_weight: float,
    dense_weight: float,
) -> List[CandidateRecord]:
    """Public wrapper so other modules can reuse the hybrid scorer."""

    return _score_query(
        query,
        model,
        chunk_embeddings,
        bm25,
        chunk_ids,
        chunk_texts,
        top_k,
        lexical_weight,
        dense_weight,
    )


def validate_positive_hits(query: QueryExample, positives_in_topk: int) -> Optional[QueryValidationIssue]:
    """Expose validation helper so downstream evaluators share the same logic."""

    return _validate_positive_hits(query, positives_in_topk)


def validate_negative_hits(query: QueryExample, negatives_in_topk: int) -> Optional[QueryValidationIssue]:
    """Ensure each candidate slate still retains at least one negative."""

    return _validate_negative_hits(query, negatives_in_topk)


def probe_test_question(
    question: str,
    model: SentenceTransformer,
    chunk_embeddings: torch.Tensor,
    bm25: BM25Lite,
    chunk_ids: Sequence[str],
    chunk_texts: Sequence[str],
    top_k: int,
    lexical_weight: float,
    dense_weight: float,
    *,
    logger: logging.Logger,
) -> None:
    if not question:
        return
    synthetic = QueryExample(
        query_id="test_question",
        text=question,
        positives=[],
        difficulty="probe",
        split="probe",
        metadata={},
    )
    candidates = _score_query(
        synthetic,
        model,
        chunk_embeddings,
        bm25,
        chunk_ids,
        chunk_texts,
        top_k,
        lexical_weight,
        dense_weight,
    )
    preview = [
        {
            "rank": cand.rank,
            "chunk_id": cand.candidate_id,
            "score": cand.score_combined,
        }
        for cand in candidates[:5]
    ]
    logger.info("Test question preview: %s", preview, extra={"Query": question[:64], "Stage": "probe"})


def build_candidate_pairs(
    queries: Sequence[QueryExample],
    *,
    chunk_map: Dict[str, Dict[str, Any]],
    model: SentenceTransformer,
    chunk_embeddings: torch.Tensor,
    bm25: BM25Lite,
    chunk_ids: Sequence[str],
    chunk_texts: Sequence[str],
    top_k: int,
    lexical_weight: float,
    dense_weight: float,
    validate_only: bool,
    logger: logging.Logger,
) -> Tuple[
    Dict[str, Dict[str, List[Dict[str, Any]]]],
    Dict[str, Dict[str, List[Dict[str, Any]]]],
    List[Dict[str, Any]],
    List[QueryValidationIssue],
]:
    pairs: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    triplets: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    query_breakdown: List[Dict[str, Any]] = []
    validation_issues: List[QueryValidationIssue] = []

    for query in queries:
        query_ctx = build_query_context(query)
        candidates = _score_query(
            query,
            model,
            chunk_embeddings,
            bm25,
            chunk_ids,
            chunk_texts,
            top_k,
            lexical_weight,
            dense_weight,
        )
        positives_in_topk = sum(cand.label for cand in candidates)
        negatives_in_topk = max(0, len(candidates) - positives_in_topk)
        positives_total = len(query.positives)
        if not positives_in_topk:
            logger.warning(
                "Query %s has no positives in top-%s",
                query.query_id,
                top_k,
                extra={"Query": query.text, "Stage": "retrieval"},
            )
        issue = _validate_positive_hits(query, positives_in_topk)
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
        neg_issue = _validate_negative_hits(query, negatives_in_topk)
        if neg_issue:
            validation_issues.append(neg_issue)
            logger.error(
                "Query %s lacks negative coverage in top-%s",
                query.query_id,
                top_k,
                extra={"Query": query.text, "Stage": "validation"},
            )
        top_rank = min((cand.rank for cand in candidates if cand.label == 1), default=None)
        query_breakdown.append(
            {
                "query_id": query.query_id,
                "difficulty": query.difficulty,
                "split": query.split,
                "positives_total": positives_total,
                "positives_in_topk": positives_in_topk,
                "negatives_in_topk": negatives_in_topk,
                "top_positive_rank": top_rank,
                "pairs": len(candidates),
            }
        )

        if validate_only:
            continue

        chunk_lookup = chunk_map
        for cand in candidates:
            candidate_ctx = build_candidate_context(query_ctx, chunk_lookup.get(cand.candidate_id), cand)
            context_payload = build_context_payload(query_ctx, candidate_ctx)
            record = {
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
                "context": context_payload,
            }
            pairs[query.difficulty][query.split].append(record)
        triplet = _build_triplet(query, candidates)
        if triplet:
            triplets[query.difficulty][query.split].append(triplet)

    return pairs, triplets, query_breakdown, validation_issues


__all__ = [
    "BM25Lite",
    "CandidateRecord",
    "QueryValidationIssue",
    "build_bm25_index",
    "build_candidate_pairs",
    "encode_corpus",
    "probe_test_question",
    "score_query_candidates",
    "tokenize_text",
    "validate_positive_hits",
]
