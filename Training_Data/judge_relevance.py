"""Judge relevance of candidate chunks for each synthetic query.

For each query in synthetic_queries_raw.json:
  1. Run hybrid retrieval (BM25 + dense) to get top-K candidate chunks.
  2. Send the query + candidates to the LLM for relevance scoring (0/1/2).
  3. Assign chunks with score >= 2 as positive_ids.
  4. Merge judged queries into retriever_eval_queries.json.

Usage:
    python judge_relevance.py              # judge all raw synthetic queries
    python judge_relevance.py --limit 10   # judge only 10 queries (testing)
    python judge_relevance.py --resume     # resume from last checkpoint
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from huggingface_hub import get_inference_endpoint

from config_training_data import (
    AUTO_PAUSE_ENDPOINT,
    CHUNKS_FILE,
    DENSE_WEIGHT,
    EMBED_MODEL_PATH,
    ENDPOINT_NAME,
    HF_TOKEN,
    JUDGE_MODEL_MAX_TOKENS,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_TEMPERATURE,
    JUDGE_TOP_K,
    LEXICAL_WEIGHT,
    QUERIES_FILE,
    RETRIEVER_TOP_K,
    SYNTHETIC_RAW_FILE,
)


def _log(message: str) -> None:
    """Always-flushed console logging for long-running stages."""
    print(message, flush=True)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_endpoint_url: Optional[str] = None
_endpoint_obj = None
_chunks_by_id: Dict[str, dict] = {}
_bm25_index = None
_embed_model = None
_chunk_embeddings = None
_chunk_ids_ordered: List[str] = []

JUDGED_CHECKPOINT_FILE = SYNTHETIC_RAW_FILE.parent / "synthetic_queries_judged.json"
RETRIEVAL_CACHE_FILE = SYNTHETIC_RAW_FILE.parent / "retrieval_dense_cache.npz"

_status_stage = "startup"
_status_detail = ""
_heartbeat_thread: Optional[threading.Thread] = None
_heartbeat_stop_event = threading.Event()


def _set_status(stage: str, detail: str = "") -> None:
    global _status_stage, _status_detail
    _status_stage = stage
    _status_detail = detail
    if detail:
        _log(f"[STATUS] {stage} :: {detail}")
    else:
        _log(f"[STATUS] {stage}")


def _heartbeat_loop(interval_seconds: int) -> None:
    tick = 0
    while not _heartbeat_stop_event.wait(interval_seconds):
        tick += 1
        detail = f" detail={_status_detail}" if _status_detail else ""
        _log(f"[HEARTBEAT] t+{tick * interval_seconds}s stage={_status_stage}{detail}")


def _start_heartbeat(interval_seconds: int) -> None:
    global _heartbeat_thread
    if interval_seconds <= 0:
        _log("[HEARTBEAT] disabled (interval <= 0)")
        return
    _heartbeat_stop_event.clear()
    _heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(interval_seconds,),
        name="judge-heartbeat",
        daemon=True,
    )
    _heartbeat_thread.start()
    _log(f"[HEARTBEAT] started (every {interval_seconds}s)")


def _stop_heartbeat() -> None:
    global _heartbeat_thread
    _heartbeat_stop_event.set()
    if _heartbeat_thread is not None:
        _heartbeat_thread.join(timeout=2)
        _heartbeat_thread = None
    _log("[HEARTBEAT] stopped")


# ---------------------------------------------------------------------------
# HF Endpoint lifecycle  (same pattern as generate_synthetic_queries.py)
# ---------------------------------------------------------------------------

def _start_endpoint() -> str:
    global _endpoint_obj
    _set_status("hf_endpoint", "checking endpoint status")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set")

    ep = get_inference_endpoint(ENDPOINT_NAME, token=HF_TOKEN)
    _endpoint_obj = ep
    status = ep.status
    _log(f"[HF] Endpoint '{ENDPOINT_NAME}' status: {status}")

    if status == "running":
        _set_status("hf_endpoint", "already running")
        _log(f"[HF] Already running at {ep.url}")
    elif status == "paused":
        _set_status("hf_endpoint", "resuming paused endpoint")
        _log(f"[HF] Resuming endpoint...")
        ep.resume()
        ep.wait(timeout=600)
        ep = get_inference_endpoint(ENDPOINT_NAME, token=HF_TOKEN)
        _endpoint_obj = ep
        _log(f"[HF] Ready at {ep.url}")
    elif status == "initializing":
        _set_status("hf_endpoint", "waiting for initialization")
        _log(f"[HF] Waiting for initialization...")
        ep.wait(timeout=600)
        ep = get_inference_endpoint(ENDPOINT_NAME, token=HF_TOKEN)
        _endpoint_obj = ep
        _log(f"[HF] Ready at {ep.url}")
    else:
        raise RuntimeError(f"Endpoint in unexpected state: {status}")
    return ep.url


def _pause_endpoint() -> None:
    if not AUTO_PAUSE_ENDPOINT:
        _log("[HF] Auto-pause disabled; endpoint remains running.")
        return
    try:
        ep = get_inference_endpoint(ENDPOINT_NAME, token=HF_TOKEN)
        ep.pause()
        _log(f"[HF] Endpoint '{ENDPOINT_NAME}' paused successfully.")
    except Exception as exc:
        _log(f"[HF] WARNING: could not pause endpoint: {exc}")


def _call_llm(system_prompt: str, user_content: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": JUDGE_MODEL_MAX_TOKENS,
        "temperature": JUDGE_TEMPERATURE,
    }
    resp = requests.post(
        f"{_endpoint_url}/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Retrieval setup  (lightweight BM25 + dense hybrid, no external deps)
# ---------------------------------------------------------------------------

def _init_retrieval(
    chunks: List[dict],
    *,
    use_dense: bool = True,
    allow_dense_cache: bool = True,
    require_dense_cache: bool = False,
) -> None:
    """Build BM25 index and load embedding model for hybrid retrieval."""
    global _bm25_index, _embed_model, _chunk_embeddings, _chunk_ids_ordered, _chunks_by_id

    _set_status("retrieval", "building BM25 and dense indices")
    _log("[RETRIEVAL] Stage start: building retrieval indices...")

    # Index chunks by id
    _chunks_by_id = {c["id"]: c for c in chunks if c.get("id")}
    _chunk_ids_ordered = [c["id"] for c in chunks if c.get("id")]

    # --- BM25 ---
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[RETRIEVAL] Installing rank_bm25...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rank-bm25", "-q"])
        from rank_bm25 import BM25Okapi

    corpus = []
    for cid in _chunk_ids_ordered:
        c = _chunks_by_id[cid]
        text = (c.get("content") or "") + " " + (c.get("code_friendly_name") or "")
        corpus.append(text.lower().split())
    _bm25_index = BM25Okapi(corpus)
    _log(f"[RETRIEVAL] BM25 index built ({len(corpus)} docs)")

    if not use_dense:
        _set_status("retrieval", "bm25-only ready")
        _log("[RETRIEVAL] Dense retrieval disabled (--lexical-only). Using BM25 only.")
        return

    # --- Dense embeddings ---
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        _log("[RETRIEVAL] Installing sentence-transformers...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "sentence-transformers", "-q"])
        from sentence_transformers import SentenceTransformer
        import numpy as np

    model_path = str(EMBED_MODEL_PATH)
    _set_status("retrieval", f"loading embedding model from {model_path}")
    _log(f"[RETRIEVAL] Loading embedding model from {model_path}")
    _embed_model = SentenceTransformer(model_path)

    # Reuse existing chunk embeddings when present; otherwise use cache or encode now.
    precomputed_embeddings = []
    for cid in _chunk_ids_ordered:
        emb = _chunks_by_id[cid].get("embedding")
        if isinstance(emb, list) and emb:
            precomputed_embeddings.append(emb)
        else:
            precomputed_embeddings = []
            break

    if precomputed_embeddings:
        _log(f"[RETRIEVAL] Using precomputed embeddings for {len(precomputed_embeddings)} chunks")
        arr = np.array(precomputed_embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _chunk_embeddings = arr / norms
    else:
        cache_loaded = False
        if allow_dense_cache and RETRIEVAL_CACHE_FILE.exists():
            _set_status("retrieval", "attempting dense cache load")
            _log(f"[RETRIEVAL] Found dense cache: {RETRIEVAL_CACHE_FILE}")
            try:
                cache = np.load(RETRIEVAL_CACHE_FILE, allow_pickle=True)
                cached_ids = cache["chunk_ids"].tolist()
                cached_embeddings = cache["embeddings"]
                if cached_ids == _chunk_ids_ordered and len(cached_embeddings) == len(_chunk_ids_ordered):
                    _chunk_embeddings = cached_embeddings.astype(np.float32)
                    cache_loaded = True
                    _log(f"[RETRIEVAL] Loaded dense cache for {len(_chunk_ids_ordered)} chunks")
                else:
                    _log("[RETRIEVAL] Dense cache mismatch (chunk IDs changed); rebuilding cache")
            except Exception as exc:
                _log(f"[RETRIEVAL] Dense cache load failed; rebuilding. Reason: {exc}")

        if require_dense_cache and not cache_loaded:
            raise RuntimeError(
                "Dense cache required but not available. Run with --prepare-only first."
            )

        if cache_loaded:
            _set_status("retrieval", "dense cache ready")
            _log("[RETRIEVAL] Dense index ready (from cache)")
            return

        texts = []
        for cid in _chunk_ids_ordered:
            c = _chunks_by_id[cid]
            texts.append(c.get("content") or "")
        batch_size = 64
        total = len(texts)
        total_batches = (total + batch_size - 1) // batch_size
        _log(f"[RETRIEVAL] Encoding {total} chunks in {total_batches} batches...")
        _set_status("retrieval", f"encoding {total} chunks in {total_batches} batches")
        vectors = []
        for batch_index, start in enumerate(range(0, total, batch_size), 1):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]
            batch_vecs = _embed_model.encode(
                batch_texts,
                show_progress_bar=False,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            vectors.append(batch_vecs)
            _log(
                f"[RETRIEVAL] Encoded batch {batch_index}/{total_batches} "
                f"(chunks {start + 1}-{end} of {total})"
            )
            _set_status("retrieval", f"encoded batch {batch_index}/{total_batches}")
        _chunk_embeddings = np.vstack(vectors)

        if allow_dense_cache:
            try:
                np.savez_compressed(
                    RETRIEVAL_CACHE_FILE,
                    chunk_ids=np.array(_chunk_ids_ordered, dtype=object),
                    embeddings=_chunk_embeddings.astype(np.float32),
                )
                _log(f"[RETRIEVAL] Saved dense cache: {RETRIEVAL_CACHE_FILE}")
            except Exception as exc:
                _log(f"[RETRIEVAL] WARNING: Failed to save dense cache: {exc}")
    _log("[RETRIEVAL] Dense index ready")
    _set_status("retrieval", "dense index ready")


def _retrieve(query: str, top_k: int = RETRIEVER_TOP_K, *, use_dense: bool = True) -> List[Tuple[str, float]]:
    """Hybrid BM25 + dense retrieval.  Returns [(chunk_id, score), ...]."""
    import numpy as np

    # BM25 scores
    tokens = query.lower().split()
    bm25_scores = _bm25_index.get_scores(tokens)

    if use_dense and _embed_model is not None and _chunk_embeddings is not None:
        # Dense scores
        q_emb = _embed_model.encode([query], normalize_embeddings=True)
        dense_scores = (_chunk_embeddings @ q_emb.T).flatten()
    else:
        dense_scores = np.zeros_like(bm25_scores)

    # Normalize each to [0, 1]
    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    bm25_norm = _norm(bm25_scores)
    dense_norm = _norm(dense_scores)

    # Hybrid (or lexical-only)
    if use_dense and _embed_model is not None and _chunk_embeddings is not None:
        combined = LEXICAL_WEIGHT * bm25_norm + DENSE_WEIGHT * dense_norm
    else:
        combined = bm25_norm

    # Top-K
    top_indices = np.argsort(combined)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append((_chunk_ids_ordered[idx], float(combined[idx])))
    return results


# ---------------------------------------------------------------------------
# Relevance parsing
# ---------------------------------------------------------------------------

def _parse_judge_response(raw: str) -> Dict[str, int]:
    """Parse LLM judge response into {chunk_id: score}."""
    import re

    text = raw.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    text = text[start : end + 1]

    # Fix trailing commas
    text = re.sub(r",\s*}", "}", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {k: int(v) for k, v in parsed.items() if isinstance(v, (int, float))}
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


# ---------------------------------------------------------------------------
# Build judge prompt
# ---------------------------------------------------------------------------

def _build_judge_user_prompt(query: str, candidates: List[Tuple[str, float]]) -> str:
    """Build user prompt with the query and top candidates for judging."""
    parts = [f"Query: {query}\n\nCandidate chunks:\n"]

    for rank, (cid, score) in enumerate(candidates[:JUDGE_TOP_K], 1):
        chunk = _chunks_by_id.get(cid, {})
        heading = chunk.get("concat_header_path") or chunk.get("heading") or ""
        content = (chunk.get("content") or "")[:800]  # truncate for context window
        code_names = chunk.get("code_friendly_name") or ""

        parts.append(f"--- Chunk {rank}: {cid} ---")
        if heading:
            parts.append(f"Section: {heading}")
        if code_names:
            parts.append(f"API names: {code_names}")
        parts.append(content)
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _load_judged_checkpoint() -> List[dict]:
    if JUDGED_CHECKPOINT_FILE.exists():
        with open(JUDGED_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            _log(f"[CHECKPOINT] Loaded {len(data)} judged queries.")
            return data
    _log("[CHECKPOINT] No existing judged checkpoint found")
    return []


def _save_judged_checkpoint(queries: List[dict]) -> None:
    with open(JUDGED_CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)


def _load_json_file(path: Path) -> list:
    """Load JSON list with UTF-8 BOM fallback and clear errors."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        if "Unexpected UTF-8 BOM" in str(exc):
            _log(f"[JSON] BOM detected in {path.name}; retrying with utf-8-sig")
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        else:
            raise

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
    return data


def _write_json_atomic(path: Path, data: list) -> None:
    """Write JSON atomically and keep a backup of the previous file."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    backup_path = path.with_suffix(path.suffix + ".bak")

    if path.exists():
        shutil.copy2(path, backup_path)
        _log(f"[WRITE] Backup created: {backup_path}")

    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    os.replace(tmp_path, path)
    _log(f"[WRITE] Atomic write complete: {path}")


# ---------------------------------------------------------------------------
# Merge into final queries file
# ---------------------------------------------------------------------------

def _merge_into_queries_file(judged: List[dict]) -> None:
    """Merge judged synthetic queries into the canonical retriever_eval_queries.json."""
    # Load existing
    if QUERIES_FILE.exists():
        existing = _load_json_file(QUERIES_FILE)
    else:
        existing = []

    # Remove any previously merged synthetic queries (idempotent)
    existing = [q for q in existing if q.get("source") != "synthetic"]

    # Append judged queries (only those with at least one positive)
    added = 0
    for q in judged:
        if q.get("positive_ids"):
            existing.append(q)
            added += 1

    # Save
    _write_json_atomic(QUERIES_FILE, existing)

    _log(f"[MERGE] Wrote {len(existing)} total queries to {QUERIES_FILE} "
         f"({added} synthetic with positives)")


# ---------------------------------------------------------------------------
# Main judging loop
# ---------------------------------------------------------------------------

def judge(
    *,
    resume: bool = False,
    limit: Optional[int] = None,
    lexical_only: bool = False,
    prepare_only: bool = False,
    resume_prepared: bool = False,
    merge_only: bool = False,
    heartbeat_seconds: int = 5,
) -> None:
    global _endpoint_url

    if prepare_only and resume_prepared:
        raise ValueError("--prepare-only and --resume-prepared cannot be used together")
    if merge_only and (prepare_only or resume_prepared or lexical_only or limit is not None):
        raise ValueError("--merge-only cannot be combined with --prepare-only/--resume-prepared/--lexical-only/--limit")

    _start_heartbeat(heartbeat_seconds)

    _log("=" * 70)
    _log("[START] judge_relevance.py")
    _log(
        f"[ARGS] resume={resume} limit={limit} lexical_only={lexical_only} "
        f"prepare_only={prepare_only} resume_prepared={resume_prepared} "
        f"merge_only={merge_only} heartbeat_seconds={heartbeat_seconds}"
    )
    _log(f"[PATH] chunks={CHUNKS_FILE}")
    _log(f"[PATH] raw_queries={SYNTHETIC_RAW_FILE}")
    _log(f"[PATH] judged_checkpoint={JUDGED_CHECKPOINT_FILE}")
    _log(f"[PATH] dense_cache={RETRIEVAL_CACHE_FILE}")
    _log(f"[PATH] output_queries={QUERIES_FILE}")
    _log("=" * 70)

    try:
        _set_status("startup", "validating inputs")
        if merge_only:
            _set_status("merge_only", "loading judged checkpoint and merging")
            judged_for_merge = _load_judged_checkpoint()
            if not judged_for_merge:
                raise RuntimeError(
                    f"No judged checkpoint found at {JUDGED_CHECKPOINT_FILE}; cannot run --merge-only"
                )
            _log(f"[MERGE] Running merge-only with {len(judged_for_merge)} judged records")
            _merge_into_queries_file(judged_for_merge)
            _set_status("done", "merge-only complete")
            _log("[DONE] merge-only completed successfully")
            return

        # Load raw synthetic queries
        if not SYNTHETIC_RAW_FILE.exists():
            _log(f"[ERROR] {SYNTHETIC_RAW_FILE} not found. Run generate_synthetic_queries.py first.")
            sys.exit(1)

        with open(SYNTHETIC_RAW_FILE, "r", encoding="utf-8") as f:
            raw_queries = json.load(f)
        _log(f"[DATA] Loaded {len(raw_queries)} raw synthetic queries")

    # Resume support
        judged = _load_judged_checkpoint() if resume else []
        already_judged_ids = {q["id"] for q in judged}
        if resume:
            _log(f"[DATA] Resume mode: {len(already_judged_ids)} IDs already judged")

        if limit:
            raw_queries = raw_queries[:limit]
            _log(f"[DATA] Applying --limit: processing first {len(raw_queries)} queries")

    # Load chunks and build retrieval indices locally FIRST to avoid billing HF
    # endpoint minutes during CPU/GPU prep.
        _set_status("local_prep", "loading chunks and building retrieval")
        _log("[STAGE] Local preparation start")
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        _log(f"[DATA] Loaded {len(chunks)} chunks")
        _init_retrieval(
            chunks,
            use_dense=not lexical_only,
            allow_dense_cache=True,
            require_dense_cache=resume_prepared and not lexical_only,
        )
        _log("[STAGE] Local preparation complete")

        if prepare_only:
            _set_status("done", "prepare-only complete")
            _log("[DONE] --prepare-only set: local prep complete, exiting before any HF call")
            return

    # Start endpoint only when local prep is done.
        _set_status("hf_phase", "starting endpoint and ping")
        _log("[STAGE] HF judging phase start")
        _endpoint_url = _start_endpoint()
        atexit.register(_pause_endpoint)

    # Warm-up ping to confirm endpoint traffic before entering query loop.
        try:
            _ = _call_llm(
                "You are a health-check assistant. Reply with exactly: ok",
                "health-check",
            )
            _log("[HF] Endpoint ping successful")
        except Exception as exc:
            _log(f"[HF] WARNING: endpoint ping failed (continuing): {exc}")

        processed = 0
        errors = 0

        for i, q_record in enumerate(raw_queries):
            qid = q_record["id"]

            if qid in already_judged_ids:
                _log(f"[SKIP] {qid} already judged")
                continue

            query_text = q_record["query"]
            source_chunk = q_record.get("source_chunk_id", "")

        # Retrieve candidates
            _set_status("query_retrieval", f"{qid} ({i + 1}/{len(raw_queries)})")
            _log(f"[QUERY] {i + 1}/{len(raw_queries)} id={qid} retrieving candidates")
            candidates = _retrieve(query_text, top_k=RETRIEVER_TOP_K, use_dense=not lexical_only)

        # Take top JUDGE_TOP_K for LLM judging
            top_candidates = candidates[:JUDGE_TOP_K]

        # Ensure source chunk is in the candidate list
            top_cids = {cid for cid, _ in top_candidates}
            if source_chunk and source_chunk not in top_cids:
                top_candidates.append((source_chunk, 0.0))

        # Build prompt and judge
            user_prompt = _build_judge_user_prompt(query_text, top_candidates)
            _set_status("query_judging", f"{qid} ({i + 1}/{len(raw_queries)})")
            _log(f"[HF] Judging query {i + 1}/{len(raw_queries)} ({qid})")

            try:
                raw_response = _call_llm(JUDGE_SYSTEM_PROMPT, user_prompt)
                scores = _parse_judge_response(raw_response)
            except Exception as exc:
                errors += 1
                _log(f"[ERROR] Query {qid}: {exc}")
                if errors > 20:
                    _log("[ABORT] Too many errors, stopping.")
                    break
                time.sleep(2)
                continue

        # Assign positive_ids: chunks with score >= 2
            positive_ids = [cid for cid, score in scores.items() if score >= 2]

        # If the LLM didn't score any chunk as 2, fall back to the source chunk
            if not positive_ids and source_chunk:
                positive_ids = [source_chunk]

        # Build the final query record
            judged_record = {
                "id": qid,
                "query": query_text,
                "difficulty": q_record.get("difficulty", "hard"),
                "split": q_record.get("split", "train"),
                "positive_ids": positive_ids,
                "min_positives": 1,
                "source": "synthetic",
                "source_chunk_id": source_chunk,
            }
            judged.append(judged_record)
            processed += 1
            _log(
                f"[QUERY] {qid} positives={len(positive_ids)} "
                f"top_pos={positive_ids[:3] if positive_ids else []}"
            )

        # Progress
            if processed % 10 == 0:
                _log(f"[PROGRESS] {processed} queries judged "
                     f"(avg positives: {sum(len(q['positive_ids']) for q in judged[-10:]) / min(10, len(judged)):.1f}), "
                     f"{errors} errors")

        # Checkpoint every 50
            if processed % 50 == 0:
                _save_judged_checkpoint(judged)
                _log(f"[CHECKPOINT] Saved {len(judged)} judged queries")

            time.sleep(0.3)

        # Final save
        _save_judged_checkpoint(judged)

        # Merge into canonical file
        _merge_into_queries_file(judged)

        _set_status("done", "judging complete")
        _log(f"\n{'='*60}")
        _log(f"Judging complete:")
        _log(f"  Queries judged:  {processed}")
        _log(f"  Errors:          {errors}")
        _log(f"  Total w/ pos:    {sum(1 for q in judged if q.get('positive_ids'))}")
        avg_pos = (sum(len(q["positive_ids"]) for q in judged) / len(judged)) if judged else 0
        _log(f"  Avg positives:   {avg_pos:.1f}")
        _log(f"  Output:          {QUERIES_FILE}")
        _log(f"{'='*60}")
    finally:
        _stop_heartbeat()


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _handle_signal(signum, frame):
    _log(f"\n[SIGNAL] Caught signal {signum}, saving checkpoint and exiting...")
    sys.exit(0)


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge relevance of synthetic queries")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--limit", type=int, default=None,
                        help="Judge only N queries (for testing)")
    parser.add_argument("--lexical-only", action="store_true",
                        help="Use BM25-only retrieval candidates (skip embedding model load)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Run local retrieval prep only, then exit before any HF endpoint call")
    parser.add_argument("--resume-prepared", action="store_true",
                        help="Require dense cache from --prepare-only and skip re-encoding chunks")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip judging and only merge existing judged checkpoint into canonical queries file")
    parser.add_argument("--heartbeat-seconds", type=int, default=5,
                        help="Print heartbeat status every N seconds (0 disables)")
    args = parser.parse_args()

    try:
        judge(
            resume=args.resume,
            limit=args.limit,
            lexical_only=args.lexical_only,
            prepare_only=args.prepare_only,
            resume_prepared=args.resume_prepared,
            merge_only=args.merge_only,
            heartbeat_seconds=args.heartbeat_seconds,
        )
    except Exception:
        _log("[FATAL] Unhandled exception in judge_relevance.py")
        _log(traceback.format_exc())
        sys.exit(1)
