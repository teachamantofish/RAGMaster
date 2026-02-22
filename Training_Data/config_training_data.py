"""Central configuration for synthetic query generation and relevance judging."""

from __future__ import annotations
from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Source data
CHUNKS_FILE = WORKSPACE_ROOT / "Data" / "framemaker" / "mif_jsx" / "a_chunks.json"

# Output
QUERIES_FILE = PROJECT_ROOT / "retriever_eval_queries.json"
SYNTHETIC_RAW_FILE = PROJECT_ROOT / "synthetic_queries_raw.json"  # intermediate before judging

# ---------------------------------------------------------------------------
# HuggingFace Inference Endpoint
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
ENDPOINT_NAME = "qwen3-next-80b-a3b-instruct-hmt"   # your HF endpoint name
AUTO_PAUSE_ENDPOINT = True  # pause endpoint on exit to stop charges

# ---------------------------------------------------------------------------
# Query generation settings
# ---------------------------------------------------------------------------
GENERATION_MODEL_MAX_TOKENS = 512       # max response tokens per LLM call
GENERATION_TEMPERATURE = 0.8            # higher = more diverse queries
QUERIES_PER_CHUNK = 2                   # how many queries to generate per chunk
TARGET_TOTAL_NEW_QUERIES = 500          # stop after reaching this many

# Chunks shorter than this (tokens) are skipped — too little content to form
# a meaningful query.
MIN_CHUNK_TOKENS_FOR_QUERY = 80

# System prompt for query generation
QUERY_GENERATION_SYSTEM_PROMPT = """\
You are a technical writer creating realistic search queries for a \
FrameMaker MIF/JSX scripting knowledge base.

Given a documentation chunk, write {n} diverse, natural-sounding questions \
that a developer would ask a search engine expecting THIS chunk to be a \
relevant answer.

Rules:
- Each question must be self-contained (no references to "the chunk" or "this section").
- Vary phrasing: mix how-to, what-is, troubleshooting, and lookup styles.
- Use specific API names, property names, or concepts that appear in the chunk.
- Questions should be 10-25 words long.
- Do NOT fabricate API names that are not in the chunk.

Return ONLY a JSON array of strings, nothing else. Example:
["How do I set column widths in a MIF table?", "What property controls table ruling style?"]
"""

# ---------------------------------------------------------------------------
# Relevance judging settings
# ---------------------------------------------------------------------------
JUDGE_MODEL_MAX_TOKENS = 512
JUDGE_TEMPERATURE = 0.1                 # low temp for consistent scoring
JUDGE_TOP_K = 20                        # how many retrieval candidates to judge per query

JUDGE_SYSTEM_PROMPT = """\
You are an expert relevance judge for a FrameMaker MIF/JSX scripting knowledge base.

Given a search query and a list of candidate document chunks, rate each chunk's \
relevance to the query.

Scoring:
  2 = Highly relevant — directly answers the query
  1 = Partially relevant — contains useful related information
  0 = Not relevant

Return ONLY a JSON object mapping chunk_id to score. Example:
{{"h2_abc123": 2, "h3_def456": 1, "h2_xyz789": 0}}
"""

# ---------------------------------------------------------------------------
# Retrieval settings (for building candidate lists during judging)
# ---------------------------------------------------------------------------
RETRIEVER_TOP_K = 200          # how many candidates to retrieve
LEXICAL_WEIGHT = 0.35          # BM25 weight in hybrid retrieval
DENSE_WEIGHT = 0.65            # dense weight in hybrid retrieval
EMBED_MODEL_PATH = WORKSPACE_ROOT / "Data" / "framemaker" / "mif_jsx" / "Qwen3Embed.6B-trained"

# ---------------------------------------------------------------------------
# Query ID numbering
# ---------------------------------------------------------------------------
# Existing hand-labeled queries go up to q227.  Synthetic queries start at
# q_synth_0001 to keep them visually distinct.
SYNTH_ID_PREFIX = "q_synth_"
SYNTH_ID_WIDTH = 4              # zero-padded: q_synth_0001

# New synthetic queries are always added as train split, hard difficulty.
SYNTH_SPLIT = "train"
SYNTH_DIFFICULTY = "hard"
