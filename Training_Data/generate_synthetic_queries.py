"""Generate synthetic training queries from documentation chunks.

Reads chunks from a_chunks.json, sends each to an HF Inference Endpoint LLM,
and collects candidate queries with the source chunk as the initial positive_id.

The output is an intermediate file (synthetic_queries_raw.json) that will be
further refined by judge_relevance.py to assign proper multi-chunk positive_ids.

Usage:
    python generate_synthetic_queries.py            # generate queries
    python generate_synthetic_queries.py --resume   # resume from last checkpoint
    python generate_synthetic_queries.py --limit 10 # process only 10 chunks (testing)
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

import requests
from huggingface_hub import get_inference_endpoint

from config_training_data import (
    AUTO_PAUSE_ENDPOINT,
    CHUNKS_FILE,
    ENDPOINT_NAME,
    GENERATION_MODEL_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    HF_TOKEN,
    MIN_CHUNK_TOKENS_FOR_QUERY,
    QUERIES_PER_CHUNK,
    QUERY_GENERATION_SYSTEM_PROMPT,
    SYNTH_DIFFICULTY,
    SYNTH_ID_PREFIX,
    SYNTH_ID_WIDTH,
    SYNTH_SPLIT,
    SYNTHETIC_RAW_FILE,
    TARGET_TOTAL_NEW_QUERIES,
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_endpoint_url: Optional[str] = None
_endpoint_obj = None  # keep reference for cleanup


# ---------------------------------------------------------------------------
# HF Endpoint lifecycle
# ---------------------------------------------------------------------------

def _start_endpoint() -> str:
    """Resume / wait-for the HF Inference Endpoint and return its URL."""
    global _endpoint_obj
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set")

    ep = get_inference_endpoint(ENDPOINT_NAME, token=HF_TOKEN)
    _endpoint_obj = ep
    status = ep.status
    print(f"[HF] Endpoint '{ENDPOINT_NAME}' status: {status}")

    if status == "running":
        print(f"[HF] Already running at {ep.url}")
    elif status == "paused":
        print(f"[HF] Resuming endpoint...")
        ep.resume()
        ep.wait(timeout=600)
        ep = get_inference_endpoint(ENDPOINT_NAME, token=HF_TOKEN)
        _endpoint_obj = ep
        print(f"[HF] Ready at {ep.url}")
    elif status == "initializing":
        print(f"[HF] Waiting for initialization...")
        ep.wait(timeout=600)
        ep = get_inference_endpoint(ENDPOINT_NAME, token=HF_TOKEN)
        _endpoint_obj = ep
        print(f"[HF] Ready at {ep.url}")
    else:
        raise RuntimeError(
            f"Endpoint in unexpected state: {status}. "
            "Check https://ui.endpoints.huggingface.co/"
        )
    return ep.url


def _pause_endpoint() -> None:
    """Pause the endpoint to stop charges."""
    if not AUTO_PAUSE_ENDPOINT:
        print("[HF] Auto-pause disabled; endpoint remains running.")
        return
    try:
        ep = get_inference_endpoint(ENDPOINT_NAME, token=HF_TOKEN)
        ep.pause()
        print(f"[HF] Endpoint '{ENDPOINT_NAME}' paused successfully.")
    except Exception as exc:
        print(f"[HF] WARNING: could not pause endpoint: {exc}")


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(system_prompt: str, user_content: str) -> str:
    """Send a chat-completion request to the HF endpoint."""
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": GENERATION_MODEL_MAX_TOKENS,
        "temperature": GENERATION_TEMPERATURE,
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
# Query extraction helpers
# ---------------------------------------------------------------------------

def _parse_query_list(raw: str) -> List[str]:
    """Extract a JSON list of strings from the LLM response.

    Handles markdown fences, trailing commas, and other common LLM quirks.
    """
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Find the JSON array portion
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    text = text[start : end + 1]

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [q.strip() for q in parsed if isinstance(q, str) and q.strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: try fixing trailing commas
    import re
    text = re.sub(r",\s*]", "]", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [q.strip() for q in parsed if isinstance(q, str) and q.strip()]
    except json.JSONDecodeError:
        pass

    return []


def _build_user_prompt(chunk: dict) -> str:
    """Build the user message containing chunk content for the LLM."""
    parts = []

    heading = chunk.get("concat_header_path") or chunk.get("heading") or ""
    if heading:
        parts.append(f"Section: {heading}")

    summary = chunk.get("chunk_summary") or ""
    if summary:
        parts.append(f"Summary: {summary}")

    content = chunk.get("content") or ""
    # Truncate very long content to ~3000 chars to stay within context
    if len(content) > 3000:
        content = content[:3000] + "\n[...truncated]"
    parts.append(f"Content:\n{content}")

    code_friendly = chunk.get("code_friendly_name") or ""
    if code_friendly:
        parts.append(f"Related API names: {code_friendly}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _load_checkpoint() -> List[dict]:
    """Load existing raw queries from the intermediate file."""
    if SYNTHETIC_RAW_FILE.exists():
        with open(SYNTHETIC_RAW_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"[CHECKPOINT] Loaded {len(data)} existing synthetic queries.")
            return data
    return []


def _save_checkpoint(queries: List[dict]) -> None:
    """Persist the current batch of synthetic queries."""
    with open(SYNTHETIC_RAW_FILE, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(*, resume: bool = False, limit: Optional[int] = None) -> None:
    global _endpoint_url

    # Load chunks
    print(f"[DATA] Loading chunks from {CHUNKS_FILE}")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[DATA] Loaded {len(chunks)} chunks")

    # Filter to chunks with enough content
    eligible = [
        c for c in chunks
        if c.get("token_count", 0) >= MIN_CHUNK_TOKENS_FOR_QUERY
        and c.get("content", "").strip()
        and c.get("id")
    ]
    print(f"[DATA] {len(eligible)} chunks eligible (>= {MIN_CHUNK_TOKENS_FOR_QUERY} tokens)")

    # Shuffle for diversity (deterministic seed for reproducibility)
    random.seed(42)
    random.shuffle(eligible)

    # Resume support
    existing = _load_checkpoint() if resume else []
    already_done_chunks = {q["source_chunk_id"] for q in existing}
    total_queries = len(existing)

    if resume and total_queries > 0:
        print(f"[RESUME] {total_queries} queries already generated, "
              f"{len(already_done_chunks)} chunks processed")

    # Start endpoint
    _endpoint_url = _start_endpoint()

    # Register cleanup to pause endpoint on any exit
    atexit.register(_pause_endpoint)

    # Build system prompt
    system_prompt = QUERY_GENERATION_SYSTEM_PROMPT.format(n=QUERIES_PER_CHUNK)

    # Track progress
    all_queries = list(existing)
    chunks_processed = 0
    errors = 0

    if limit:
        eligible = eligible[:limit]

    for i, chunk in enumerate(eligible):
        chunk_id = chunk["id"]

        # Skip already-processed chunks (resume mode)
        if chunk_id in already_done_chunks:
            continue

        # Check target
        if total_queries >= TARGET_TOTAL_NEW_QUERIES:
            print(f"[DONE] Reached target of {TARGET_TOTAL_NEW_QUERIES} queries.")
            break

        # Build prompt and call LLM
        user_prompt = _build_user_prompt(chunk)

        try:
            raw_response = _call_llm(system_prompt, user_prompt)
            queries = _parse_query_list(raw_response)
        except Exception as exc:
            errors += 1
            print(f"[ERROR] Chunk {chunk_id}: {exc}")
            if errors > 20:
                print("[ABORT] Too many errors, stopping.")
                break
            time.sleep(2)
            continue

        if not queries:
            print(f"[WARN] Chunk {chunk_id}: no queries parsed from response")
            errors += 1
            continue

        # Create query records
        for q_text in queries[:QUERIES_PER_CHUNK]:
            total_queries += 1
            seq = str(total_queries).zfill(SYNTH_ID_WIDTH)
            record = {
                "id": f"{SYNTH_ID_PREFIX}{seq}",
                "query": q_text,
                "difficulty": SYNTH_DIFFICULTY,
                "split": SYNTH_SPLIT,
                "positive_ids": [chunk_id],
                "min_positives": 1,
                "source": "synthetic",
                "source_chunk_id": chunk_id,
            }
            all_queries.append(record)

        chunks_processed += 1

        # Progress logging
        if chunks_processed % 10 == 0:
            print(f"[PROGRESS] {chunks_processed} chunks → {total_queries} queries "
                  f"({errors} errors)")

        # Checkpoint every 50 chunks
        if chunks_processed % 50 == 0:
            _save_checkpoint(all_queries)
            print(f"[CHECKPOINT] Saved {total_queries} queries")

        # Brief pause to avoid rate limits
        time.sleep(0.3)

    # Final save
    _save_checkpoint(all_queries)
    print(f"\n{'='*60}")
    print(f"Generation complete:")
    print(f"  Chunks processed: {chunks_processed}")
    print(f"  Total queries:    {total_queries}")
    print(f"  Errors:           {errors}")
    print(f"  Output:           {SYNTHETIC_RAW_FILE}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Signal handling for graceful shutdown
# ---------------------------------------------------------------------------

def _handle_signal(signum, frame):
    print(f"\n[SIGNAL] Caught signal {signum}, saving checkpoint and exiting...")
    # atexit will handle endpoint pause
    sys.exit(0)


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic queries from chunks")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only N chunks (for testing)")
    args = parser.parse_args()

    generate(resume=args.resume, limit=args.limit)
