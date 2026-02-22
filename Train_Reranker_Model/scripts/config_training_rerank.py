"""Central configuration for the reranker-only pipeline."""

from __future__ import annotations
from pathlib import Path
import os

# Base workspace and logging -------------------------------------------------
BASE_CWD = Path("C:/GIT/Z_Master_Rag/Data/framemaker/mif_jsx")
LOG_FILES = BASE_CWD  # Individual scripts can override if needed.
MASTER_RERANK_LOG = "master_rerank_training_log.csv"
CODE_CHANGE = "Exp39: DeBERTa-v3-base (86M) + expanded synthetic query training data (~500 new queries)"  # Optional short description of code changes for logging purposes.

# Directory naming for reranker artifacts ------------------------------------
RERANK_TRAINING_SUBDIR = "reranker_training_data"
RERANK_OUTPUT_SUBDIR = "DeBERTa-v3-base86M-trained"
RETRIEVER_BASELINE_SUBDIR = "Qwen3Embed.6B-trained"  # existing retriever export
CROSS_ENCODER_PAIR_SUBDIR = "cross_encoder_pairs"
# Historical embed exports that get merged into RERANK_TRAINING_SUBDIR.
EMBED_SOURCE_SUBDIRS = ("embedding_training_data", "embedding_training_data1")

# TRAINING_DATA_DIR now hosts the merged embed triplets. We still keep the
# reranker-specific create-data script for refinements, but its default mode is
# to reuse this directory and only emit cross-encoder pairs.
TRAINING_DATA_DIR = BASE_CWD / RERANK_TRAINING_SUBDIR
CROSS_ENCODER_DATA_DIR = TRAINING_DATA_DIR / CROSS_ENCODER_PAIR_SUBDIR
RERANKER_OUTPUT_PATH = BASE_CWD / RERANK_OUTPUT_SUBDIR
RETRIEVER_BASELINE_MODEL_PATH = BASE_CWD / RETRIEVER_BASELINE_SUBDIR

# Model definition -----------------------------------------------------------
BASE_MODEL = "microsoft/deberta-v3-base"
CONFIG_MODEL_NAME = BASE_MODEL

# Allow optional comma-separated overrides for ad-hoc experiments.
_default_doclist = os.environ.get("RERANK_DOCLIST", RERANK_TRAINING_SUBDIR)
DOCLIST = [part.strip() for part in _default_doclist.split(",") if part.strip()]
if not DOCLIST:
    DOCLIST = [RERANK_TRAINING_SUBDIR]

# Convenience mapping so orchestration scripts know where to read/write.
RERANK_DATA_SUBDIRS = {
    "triplets": RERANK_TRAINING_SUBDIR,
    "cross_encoder_pairs": CROSS_ENCODER_PAIR_SUBDIR,
    "retriever_baseline_model": RETRIEVER_BASELINE_SUBDIR,
}

# ---------------------------------------------------------------------------
# Data generation controls (used when --regenerate-triplets is requested)
RERANK_INSTRUCTION = (
    "Given a FrameMaker MIF or JSX scripting question, retrieve information about MIF object structure and JSX "
    "script behavior."
)
MIN_CONTENT_LENGTH = 100
MAX_CONTENT_LENGTH = 4000
MIN_TRIPLETS_PER_CATEGORY = 200
MAX_TRIPLETS_PER_CATEGORY = 400
TRAIN_TEST_SPLIT = 0.8
NEGATIVE_SAMPLING_RATIO = 2
GENERATE_DIFFICULTY_LEVELS = True
ANCHOR_CATEGORY_BALANCE = {"jsx": 0.5, "mif": 0.5}
SAME_SOURCE_POSITIVE_RATIO = 0.7
CROSS_SOURCE_HARD_NEGATIVE_RATIO = 0.3
EASY_TRIPLET_RATIO = 0.20
MEDIUM_TRIPLET_RATIO = 0.35
HARD_TRIPLET_RATIO = 0.45

# Cross-encoder training hyperparameters ------------------------------------
RERANKER_TRAINING_CONFIG = {
    "epochs": 3,
    "learning_rate": 2e-5,
    "batch_size": 32,
    "gradient_accumulation_steps": 2,  # NOTE: not supported by CrossEncoder.fit()
    "warmup_steps": 50,
    "max_length": 512,
    "use_amp": True,
    "loss": "binary_ce",
    "freeze_base": False,
}

# Hybrid retrieval + rerank defaults ----------------------------------------
HYBRID_PIPELINE_CONFIG = {
    "retriever_top_k": 200,
    "reranker_top_k": 20,
    "reranker_batch_size": 64,
    "retriever_shortlist_field": "candidate_chunks",
    "score_field": "reranker_score",
}

# Retriever-backed candidate generation defaults ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent  # C:/GIT/Z_Master_Rag

RETRIEVER_PIPELINE_CONFIG = {
    "queries_file": WORKSPACE_ROOT / "Training_Data" / "retriever_eval_queries.json",
    "chunks_file": BASE_CWD / "a_chunks.json",
    "top_k": 200,
    "lexical_weight": 0.35,
    "dense_weight": 0.65,
    "max_queries": None,
    "test_question": "How do I convert a FrameMaker table to MIF via JSX?",
    "log_filename": "a_build_retrieval_candidates.log",
    "min_positive_hits": 1,
    "min_ground_truth": 1,
}

# Score Interpolation defaults -----------------------------------------------
# Sweep alpha values: final_score = α·reranker + (1−α)·baseline
# alpha=1.0 is pure reranker, alpha=0.0 is pure baseline.
# Set to empty list [] to skip the sweep.
INTERPOLATION_ALPHAS = [0.60, 0.63, 0.65, 0.67, 0.70, 0.73, 0.75, 0.77, 0.80]

# Evaluation/reporting defaults ---------------------------------------------
# Notes:
#   - difficulties: list of difficulty labels (easy|medium|hard) that match query definitions.
#   - default_split: train or test (must match split values in retriever_eval_queries.json).
#   - use_live_retrieval: True to rebuild candidate slates each run, False to use prebuilt pairs.
#   - allow_missing_ground_truth / allow_missed_positives: set True to keep going when
#     query definitions resolve too few positives or retrieval coverage is insufficient.
RERANK_EVAL_CONFIG = {
    # difficulties: ordered subset drawn from ["easy", "medium", "hard"]
    "difficulties": ["easy", "medium", "hard"],
    # default_split accepts "train", "test", or "all" (evaluate both splits)
    "default_split": "all",
    "use_live_retrieval": True,
    "allow_missing_ground_truth": False,
    "allow_missed_positives": True,
    "metrics": [
        "mrr",
        "ndcg@10",
        "recall@5",
        "recall@10",
        "map@10",
    ],
    "emit_csv": True,
    "emit_json": True,
    "comparison_baseline": "retriever_only",
}
