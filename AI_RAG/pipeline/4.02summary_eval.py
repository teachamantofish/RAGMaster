# Summary Quality Evaluation Script
# Evaluates chunk summaries using BERTScore (semantic similarity) and SummaC (faithfulness)
# Usage: python 4summary_eval.py. Use the venv with the required transformers version.

# Note: Install these libraries if not already installed:
# pip install bert-score summac

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from config.summaryevalconfig import *
from common.utils import get_csv_to_process, setup_global_logger


def log_eval_settings(active_logger: Optional[object] = None) -> None:
    """Print and log configured evaluator settings for quick visibility."""
    bert_line = (
        f"  BERTScore → model:{BERT_MODEL_TYPE} lang:{BERT_LANG} "
        f"rescale:{BERT_RESCALE_WITH_BASELINE}"
    )
    summac_line = (
        "  SummaC   → model:{} granularity:{} evidence:{} primary:{}".format(
            SUMMAC_MODEL_TYPE,
            SUMMAC_GRANULARITY,
            SUMMAC_EVIDENCE_SCOPE,
            SUMMAC_PRIMARY_AGGREGATE,
        )
    )
    print("[INFO] Summary evaluation config:")
    print(bert_line)
    print(summac_line)
    if active_logger:
        active_logger.info(bert_line)
        active_logger.info(summac_line)


def _reduce_axis(values: np.ndarray, op: str) -> np.ndarray:
    if op == "mean":
        return np.mean(values, axis=0)
    if op == "min":
        return np.min(values, axis=0)
    if op == "max":
        return np.max(values, axis=0)
    return np.max(values, axis=0)


def extract_sentence_scores(image: np.ndarray, summac_model: object) -> np.ndarray:
    if image is None or image.size == 0:
        return np.array([])
    op1 = getattr(summac_model, "op1", "max")
    use_ent = getattr(summac_model, "use_ent", True)
    use_con = getattr(summac_model, "use_con", True)
    ent_layer = image[0]
    con_layer = image[1]
    ent_scores = _reduce_axis(ent_layer, op1)
    con_scores = _reduce_axis(con_layer, op1)
    if use_ent and use_con:
        sentence_scores = ent_scores - con_scores
    elif use_ent:
        sentence_scores = ent_scores
    elif use_con:
        sentence_scores = 1.0 - con_scores
    else:
        sentence_scores = ent_scores
    return np.array(sentence_scores, dtype=float)


def trimmed_mean(values: np.ndarray, proportion_to_cut: float) -> float:
    if values.size == 0:
        return float("nan")
    if proportion_to_cut <= 0:
        return float(np.mean(values))
    proportion_to_cut = min(max(proportion_to_cut, 0.0), 0.5)
    if proportion_to_cut == 0.5:
        return float(np.median(values))
    values_sorted = np.sort(values)
    cut = int(len(values_sorted) * proportion_to_cut)
    if cut == 0:
        return float(np.mean(values_sorted))
    if cut * 2 >= len(values_sorted):
        return float(np.mean(values_sorted))
    trimmed = values_sorted[cut:-cut]
    return float(np.mean(trimmed))


def aggregate_summac_scores(sentence_scores: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if sentence_scores.size == 0:
        return metrics
    metrics["mean"] = float(np.mean(sentence_scores))
    percentile = np.percentile(sentence_scores, SUMMAC_PERCENTILE * 100)
    metrics["percentile"] = float(percentile)
    metrics["trimmed_mean"] = float(trimmed_mean(sentence_scores, SUMMAC_TRIM_RATIO))
    return metrics


try:
    from bert_score import BERTScorer
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("[WARNING] bert-score not installed. Run: pip install bert-score")
    BERTSCORE_AVAILABLE = False
    exit(1)

from transformers import AutoModel

SUMMAC_RUNTIME_AVAILABLE = False
if SUMMAC_AVAILABLE:
    try:
        from summac.model_summac import SummaCZS
        SUMMAC_RUNTIME_AVAILABLE = True
    except ImportError as summac_err:
        print(
            f"[INFO] SummaC evaluation disabled (import failed): {summac_err}. "
            "Install summac or pin transformers<=4.34 to enable."
        )
else:
    print("[INFO] SummaC evaluation disabled via config (SUMMAC_AVAILABLE=False)")

chunkfile = get_csv_to_process()['cwd'] / "a_chunks.json"

# Set up global logger with script-specific CSV header
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "Chunk ID", "BERTScore F1", "SummaC Score"]
logger = setup_global_logger(script_name=script_base, log_level='INFO', headers=LOG_HEADER)
log_eval_settings(logger)

# Allow config to override Hugging Face model kwargs; default disables the unused pooler so
# transformers stops reinitializing those weights (and spamming the log) per evaluation call.
if "BERT_MODEL_KWARGS" not in globals():
    BERT_MODEL_KWARGS = {"add_pooling_layer": False}


def _patch_automodel_loader() -> None:
    """Ensure AutoModel.from_pretrained always sees our custom kwargs."""
    if hasattr(_patch_automodel_loader, "_patched"):
        return
    original_loader = AutoModel.from_pretrained

    def _patched_from_pretrained(*args, **kwargs):
        merged = dict(kwargs)
        if isinstance(BERT_MODEL_KWARGS, dict):
            for key, value in BERT_MODEL_KWARGS.items():
                merged.setdefault(key, value)
        return original_loader(*args, **merged)

    AutoModel.from_pretrained = _patched_from_pretrained  # type: ignore[method-assign]
    _patch_automodel_loader._patched = True  # type: ignore[attr-defined]


if BERTSCORE_AVAILABLE:
    _patch_automodel_loader()


def get_bertscorer() -> "BERTScorer":
    """Lazily initialize and cache a BERTScorer instance."""
    if not hasattr(get_bertscorer, "scorer"):
        print("Loading BERTScore model (first time only)...")
        get_bertscorer.scorer = BERTScorer(
            model_type=BERT_MODEL_TYPE,
            lang=BERT_LANG,
            rescale_with_baseline=BERT_RESCALE_WITH_BASELINE,
        )
    return get_bertscorer.scorer


def evaluate_summary_quality(content: str, summary: str, chunk_id: str) -> Dict[str, float]:
    """
    Evaluate summary quality using multiple metrics.
    
    Args:
        content: Original chunk content
        summary: Generated summary
        chunk_id: Chunk identifier for logging
    
    Returns:
        Dict with evaluation scores
    """
    results = {
        "chunk_id": chunk_id,
        "bert_score_f1": None,
        "summac_score": None,
        "summac_details": None,
    }
    
    # Skip evaluation if summary or content is empty
    if not summary or not content or summary.startswith("[") and "ERROR" in summary:
        logger.warning(f"Skipping evaluation for {chunk_id}: Empty or error summary")
        return results
    
    # BERTScore - Semantic similarity
    if BERTSCORE_AVAILABLE:
        try:
            scorer = get_bertscorer()
            P, R, F1 = scorer.score([summary], [content])
            results["bert_score_f1"] = float(F1[0])
            print(f"  BERTScore F1: {results['bert_score_f1']:.4f}")
        except Exception as e:
            logger.error(f"BERTScore error for {chunk_id}: {e}")
    
    # SummaC - Faithfulness/Factuality: higher scores reduce hallucinations
    if SUMMAC_RUNTIME_AVAILABLE:
        try:
            # Initialize SummaC model (cached after first use)
            if not hasattr(evaluate_summary_quality, 'summac_model'):
                print("Loading SummaC model (first time only)...")
                evaluate_summary_quality.summac_model = SummaCZS(
                    granularity=SUMMAC_GRANULARITY,
                    model_name=SUMMAC_MODEL_TYPE
                )
            
            model = evaluate_summary_quality.summac_model
            score_dict = model.score([content], [summary])
            images = score_dict.get("images") or []
            image = images[0] if images else None
            sentence_scores = extract_sentence_scores(image, model)
            aggregations = aggregate_summac_scores(sentence_scores)
            if not aggregations and score_dict.get("scores"):
                aggregations = {"mean": float(score_dict['scores'][0])}
            results["summac_details"] = aggregations or None
            if aggregations:
                primary = aggregations.get(SUMMAC_PRIMARY_AGGREGATE) or aggregations.get("mean")
                results["summac_score"] = primary
                print("  SummaC sentence metrics:")
                for key in ["mean", "percentile", "trimmed_mean"]:
                    if key in aggregations:
                        print(f"    {key}: {aggregations[key]:.4f}")
        except Exception as e:
            logger.error(f"SummaC error for {chunk_id}: {e}")
    
    return results


def main():
    """Main evaluation function."""
    print(f"\n{'='*60}")
    print("Summary Quality Evaluation")
    print(f"{'='*60}\n")
    
    # Check if required libraries are available
    if not BERTSCORE_AVAILABLE and not SUMMAC_AVAILABLE:
        print("[ERROR] No evaluation libraries available. Please install:")
        print("  pip install bert-score summac")
        return
    
    # Load chunks
    if not chunkfile.exists():
        print(f"[ERROR] Chunks file not found: {chunkfile}")
        return
    
    with open(chunkfile, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks from {chunkfile}\n")
    
    # Evaluate each chunk
    all_results = []
    chunks_evaluated = 0
    
    for i, chunk in enumerate(chunks, 1):
        chunk_id = chunk.get("id", f"unknown_{i}")
        content = chunk.get("content", "")
        summary = chunk.get("chunk_summary", "")
        
        if not summary:
            print(f"[{i}/{len(chunks)}] Skipping {chunk_id}: No summary")
            continue
        
        print(f"\n[{i}/{len(chunks)}] Evaluating {chunk_id}")
        
        results = evaluate_summary_quality(content, summary, chunk_id)
        all_results.append(results)
        
        # Log to CSV
        try:
            logger.info(
                f"Evaluated {chunk_id}",
                extra={
                    "Chunk ID": chunk_id,
                    "BERTScore F1": (
                        f"{results['bert_score_f1']:.4f}" if results['bert_score_f1'] is not None else "N/A"
                    ),
                    "SummaC Score": (
                        f"{results['summac_score']:.4f}" if results['summac_score'] is not None else "N/A"
                    ),
                }
            )
        except Exception as e:
            logger.exception(f"Failed to log results for {chunk_id}: {e}")
        
        chunks_evaluated += 1
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}\n")
    print(f"Total chunks: {len(chunks)}")
    print(f"Chunks evaluated: {chunks_evaluated}")
    
    # Calculate averages
    bert_scores = [r["bert_score_f1"] for r in all_results if r["bert_score_f1"] is not None]
    summac_scores = [r["summac_score"] for r in all_results if r["summac_score"] is not None]
    if bert_scores:
        print(f"\nBERTScore F1:")
        print(f"  Average: {sum(bert_scores) / len(bert_scores):.4f}")
        print(f"  Min: {min(bert_scores):.4f}")
        print(f"  Max: {max(bert_scores):.4f}")
    
    if summac_scores:
        print(f"\nSummaC (Faithfulness):")
        print(f"  Average: {sum(summac_scores) / len(summac_scores):.4f}")
        print(f"  Min: {min(summac_scores):.4f}")
        print(f"  Max: {max(summac_scores):.4f}")
    
    print(f"\n{'='*60}")
    if logger.handlers:
        for handler in logger.handlers:
            if hasattr(handler, 'baseFilename'):
                print(f"Results logged to: {handler.baseFilename}")
                break
        else:
            print(f"Results logged to terminal (no file handler)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")
