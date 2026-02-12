from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

from build_query_definition import QueryExample

if TYPE_CHECKING:
    from create_chunk_candidate_list import CandidateRecord


@dataclass
class QueryContext:
    query_id: str
    text: str
    difficulty: str
    split: str
    metadata: Dict[str, Any]
    positives: Any
    positive_sources: Dict[str, Any]


@dataclass
class CandidateContext:
    candidate_id: str
    text: str
    rank: int
    label: int
    scores: Dict[str, float]
    metadata: Dict[str, Any]


def build_query_context(raw_query: QueryExample, chunk_map: Optional[Dict[str, Dict[str, Any]]] = None) -> QueryContext:
    _ = chunk_map  # reserved for future metadata enrichments
    return QueryContext(
        query_id=raw_query.query_id,
        text=raw_query.text,
        difficulty=raw_query.difficulty,
        split=raw_query.split,
        metadata=raw_query.metadata,
        positives=raw_query.positives,
        positive_sources=raw_query.positive_sources,
    )


def build_candidate_context(
    query_ctx: QueryContext,
    candidate_chunk: Optional[Dict[str, Any]],
    candidate_record: "CandidateRecord",
) -> CandidateContext:
    chunk_metadata = candidate_chunk or {}
    return CandidateContext(
        candidate_id=candidate_record.candidate_id,
        text=candidate_record.text,
        rank=candidate_record.rank,
        label=candidate_record.label,
        scores={
            "dense": candidate_record.score_dense,
            "lexical": candidate_record.score_lexical,
            "combined": candidate_record.score_combined,
        },
        metadata={
            "chunk": chunk_metadata,
            "query": {
                "difficulty": query_ctx.difficulty,
                "split": query_ctx.split,
            },
        },
    )


def build_context_payload(query_ctx: QueryContext, candidate_ctx: CandidateContext) -> Dict[str, Any]:
    return {
        "query": asdict(query_ctx),
        "candidate": asdict(candidate_ctx),
    }


def format_llm_bundle(query_ctx: QueryContext, candidate_ctx: CandidateContext) -> Dict[str, Any]:
    """Placeholder for future LLM-specific payloads."""

    return build_context_payload(query_ctx, candidate_ctx)


__all__ = [
    "CandidateContext",
    "QueryContext",
    "build_candidate_context",
    "build_context_payload",
    "build_query_context",
    "format_llm_bundle",
]
