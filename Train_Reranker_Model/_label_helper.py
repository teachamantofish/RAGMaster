"""
Exp 33 labeling helper: For each excluded filter query, show:
  - The query text
  - Current filter rules
  - Which chunks match the filters (with content preview)
  - Top reranker-scored candidates (live-scored against current model)
  - Suggested best chunk IDs to pick

Output: label_candidates.md  (human-readable review file)
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict

# Allow imports from parent package
sys.path.insert(0, str(Path(__file__).parent))

from sentence_transformers import CrossEncoder
from scripts.config_training_rerank import (
    RERANKER_OUTPUT_PATH,
    RERANK_EVAL_CONFIG,
    RETRIEVER_PIPELINE_CONFIG,
    CONFIG_MODEL_NAME,
    RETRIEVER_BASELINE_MODEL_PATH,
)
from build_query_definition import load_queries
from create_chunk_candidate_list import (
    build_bm25_index,
    encode_corpus,
    score_query_candidates,
)
import chunks_loader

# ── paths ──────────────────────────────────────────────────────────────
QUERIES_FILE  = Path(__file__).parent / "retriever_eval_queries.json"
CHUNKS_FILE   = Path(r"C:\GIT\Z_Master_Rag\Data\framemaker\mif_jsx\a_chunks.json")
OUTPUT_FILE   = Path(__file__).parent / "label_candidates.md"

# ── load chunks for preview ───────────────────────────────────────────
with open(CHUNKS_FILE, encoding="utf-8") as f:
    chunks_raw = json.load(f)
    if isinstance(chunks_raw, list):
        chunks_by_id = {c["id"]: c for c in chunks_raw}
    else:
        chunks_by_id = chunks_raw

# ── load queries (only excluded ones) ─────────────────────────────────
with open(QUERIES_FILE) as f:
    all_queries_raw = json.load(f)
excluded_raw = [q for q in all_queries_raw if q.get("split") == "excluded"]

# Temporarily set split to "train" so load_queries picks them up
for q in all_queries_raw:
    if q.get("split") == "excluded":
        q["_orig_split"] = "excluded"
        q["split"] = "train"

# Write temp file for load_queries
TEMP_QUERIES = Path(__file__).parent / "_temp_queries_for_scoring.json"
with open(TEMP_QUERIES, "w") as f:
    json.dump(all_queries_raw, f, indent=2)

print(f"Loading {len(excluded_raw)} excluded queries for scoring...")

# ── load models and embeddings ────────────────────────────────────────
from sentence_transformers import SentenceTransformer
print(f"Loading baseline model: {RETRIEVER_BASELINE_MODEL_PATH}")
baseline_model = SentenceTransformer(str(RETRIEVER_BASELINE_MODEL_PATH))

# Resolve reranker model path (same logic as 3evaluate_model.py)
reranker_path = str(RERANKER_OUTPUT_PATH)
if not (Path(reranker_path) / "config.json").exists():
    print(f"No checkpoint at {reranker_path}, falling back to base model")
    reranker_path = CONFIG_MODEL_NAME
print(f"Loading reranker model from: {reranker_path}")
reranker_model = CrossEncoder(reranker_path, num_labels=1)

print("Loading chunks...")
import logging
_logger = logging.getLogger("label_helper")
chunk_list, chunk_map = chunks_loader.load_chunks(RETRIEVER_PIPELINE_CONFIG, logger=_logger)
chunk_ids = [chunk["id"] for chunk in chunk_list]
chunk_texts = [chunk["text"] for chunk in chunk_list]

print("Building BM25 index...")
bm25 = build_bm25_index(chunk_list)

print("Encoding corpus embeddings...")
chunk_embeddings = encode_corpus(baseline_model, chunk_list, batch_size=64)

# ── load and score excluded queries ───────────────────────────────────
queries, _skipped = load_queries(
    Path(TEMP_QUERIES),
    chunk_map,
    default_min_positives=0,
    default_min_positive_hits=0,
    fail_on_missing_ground_truth=False,
    logger=_logger,
)
excluded_queries = [q for q in queries if q.query_id in {eq["id"] for eq in excluded_raw}]
print(f"Loaded {len(excluded_queries)} queries, scoring against {len(chunk_ids)} chunks...")

retrieval_top_k = 200
lexical_weight = RETRIEVER_PIPELINE_CONFIG.get("lexical_weight", 0.35)
dense_weight = RETRIEVER_PIPELINE_CONFIG.get("dense_weight", 0.65)

query_candidates = {}
for qi, query in enumerate(excluded_queries, 1):
    print(f"  Scoring {qi}/{len(excluded_queries)}: {query.query_id}", end="\r")
    candidates = score_query_candidates(
        query, baseline_model, chunk_embeddings, bm25,
        chunk_ids, chunk_texts, retrieval_top_k,
        lexical_weight, dense_weight,
    )
    # Score with reranker
    pairs_for_reranker = [[query.text, c.text] for c in candidates]
    batch_size = 64
    reranker_scores = []
    for i in range(0, len(pairs_for_reranker), batch_size):
        batch = pairs_for_reranker[i:i+batch_size]
        scores = reranker_model.predict(batch)
        reranker_scores.extend(scores.tolist())

    scored = []
    for c, rscore in zip(candidates, reranker_scores):
        scored.append({
            "chunk_id": c.candidate_id,
            "reranker_score": rscore,
            "baseline_score": c.score_combined,
            "label": c.label,
        })
    scored.sort(key=lambda x: x["reranker_score"], reverse=True)
    query_candidates[query.query_id] = scored

print(f"\nScoring complete.")

# Clean up temp file
TEMP_QUERIES.unlink(missing_ok=True)

# ── resolve filters (for showing which chunks the filter matched) ─────
def matches_filter(chunk, filt):
    fields = filt.get("fields", ["content"])
    contains_any = filt.get("contains_any", [])
    contains_all = filt.get("contains_all", [])
    combined = " ".join(str(chunk.get(f, "")) for f in fields).lower()
    if contains_all:
        if not all(kw.lower() in combined for kw in contains_all):
            return False
    if contains_any:
        if not any(kw.lower() in combined for kw in contains_any):
            return False
    return bool(contains_any or contains_all)

def resolve_filters(query_raw):
    filters = query_raw.get("positive_filters", [])
    if not filters:
        return []
    matched = {}
    for filt in filters:
        max_matches = filt.get("max_matches", 999)
        count = 0
        for cid, chunk in chunks_by_id.items():
            if cid in matched:
                continue
            if matches_filter(chunk, filt):
                matched[cid] = chunk
                count += 1
                if count >= max_matches:
                    break
    return list(matched.items())

def preview(text, max_len=150):
    if not text:
        return "(empty)"
    text = re.sub(r'\s+', ' ', str(text)).strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text

# ── generate report ───────────────────────────────────────────────────
lines = []
lines.append("# Exp 33 - Label Candidates for Excluded Filter Queries\n")
lines.append("For each query, review the candidates and write the best 1-3 chunk IDs")
lines.append("into the `positive_ids` field. Then we'll update retriever_eval_queries.json.\n")
lines.append("---\n")

for qi, q in enumerate(excluded_raw, 1):
    qid = q["id"]
    qtext = q["query"]

    lines.append(f"## {qi}. {qid}: {qtext}\n")

    # Show filters
    filters = q.get("positive_filters", [])
    lines.append("**Filters:**")
    for fi, filt in enumerate(filters):
        lines.append(f"  - Filter {fi+1}: fields={filt.get('fields')}, "
                     f"contains_any={filt.get('contains_any', [])}, "
                     f"contains_all={filt.get('contains_all', [])}, "
                     f"max_matches={filt.get('max_matches', 'unlimited')}")
    lines.append("")

    # Resolve filter matches
    filter_matches = resolve_filters(q)
    filter_ids = {cid for cid, _ in filter_matches}

    # Top reranker-scored candidates
    scored = query_candidates.get(qid, [])
    lines.append(f"**Top 15 reranker-scored candidates** (filter match marked with [F]):\n")
    lines.append("| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |")
    lines.append("|------|----------|-------|-------|-----|---------|-----------------|")
    for ri, rec in enumerate(scored[:15], 1):
        cid = rec["chunk_id"]
        rscore = rec["reranker_score"]
        label = rec["label"]
        is_filter = "[F]" if cid in filter_ids else ""
        chunk = chunks_by_id.get(cid, {})
        heading = preview(chunk.get("heading", ""), 50)
        content = preview(chunk.get("content", ""), 120)
        lines.append(f"| {ri} | `{cid}` | {rscore:.2f} | {label} | {is_filter} | {heading} | {content} |")
    lines.append("")

    # Also show filter matches NOT in top 15 (might be good picks missed by retrieval)
    top15_ids = {rec["chunk_id"] for rec in scored[:15]}
    missed_filter = [(cid, chunk) for cid, chunk in filter_matches if cid not in top15_ids]
    if missed_filter:
        lines.append(f"**Filter matches NOT in top 15** ({len(missed_filter)} chunks):\n")
        lines.append("| # | Chunk ID | Heading | Content Preview |")
        lines.append("|---|----------|---------|-----------------|")
        for mi, (cid, chunk) in enumerate(missed_filter[:10], 1):
            heading = preview(chunk.get("heading", ""), 50)
            content = preview(chunk.get("content", ""), 120)
            lines.append(f"| {mi} | `{cid}` | {heading} | {content} |")
        if len(missed_filter) > 10:
            lines.append(f"| ... | ({len(missed_filter) - 10} more) | | |")
        lines.append("")

    # Decision box
    lines.append("**YOUR PICK (write 1-3 chunk IDs):**")
    lines.append("```")
    lines.append(f"positive_ids: []")
    lines.append("```")
    lines.append("\n---\n")

# ── write output ──────────────────────────────────────────────────────
OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
print(f"\nWrote {len(lines)} lines to {OUTPUT_FILE}")
print(f"Open label_candidates.md and pick the best IDs for each query.")
