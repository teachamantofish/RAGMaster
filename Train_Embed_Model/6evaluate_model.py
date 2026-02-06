"""
Evaluate Trained Embedding Models
==================================

Simple script to test and compare embedding models (epoch 1, 2, 3, or base model).

WHAT THIS DOES:
- Loads a trained embedding model
- Runs evaluation on test data
- Calculates similarity scores for anchor-positive vs anchor-negative pairs
- Reports accuracy: how often positive is closer than negative

USAGE:
    python 7evaluate_model.py --epoch 3          # Test epoch 3 model
    python 7evaluate_model.py --epoch 2          # Test epoch 2 model
    python 7evaluate_model.py --epoch 1          # Test epoch 1 model
    python 7evaluate_model.py --epoch base       # Test original base model
    python 7evaluate_model.py --all              # Test all models and compare

WHAT IT REPORTS:
- Mean similarity between anchors and positives (should be high, close to 1.0)
- Mean similarity between anchors and negatives (should be low, close to 0.0)
- Accuracy: % of cases where positive is more similar than negative
- Margin: Average difference between positive and negative similarities
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from scripts.config_training_embed import *
from scripts.custom_logger import setup_global_logger

# Set up custom logger with CSV output to LOG_FILES directory
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "Test Step", "Result"]
logger = setup_global_logger(script_name=script_base, cwd=LOG_FILES, log_level='INFO', headers=LOG_HEADER)


def get_model_path(epoch):
    """Get the path to a model based on epoch number or 'base'."""
    if epoch == "base":
        return BASE_MODEL
    elif epoch in {1, 2, 3}:
        return str(OUTPUT_MODEL_PATH)
    else:
        raise ValueError(f"Invalid epoch: {epoch}. Must be 1, 2, 3, or 'base'")


def load_test_data(difficulty: str = "hard") -> Tuple[List[Dict], str]:
    """Load test triplets from the specified difficulty level."""
    test_path = TRAINING_DATA_DIR / difficulty / "triplets_test.json"
    
    if not test_path.exists():
        print(f"⚠️  Test file not found: {test_path}")
        print(f"   Trying all difficulty levels...")
        
        # Try all difficulties
        for diff in ["hard", "medium", "easy"]:
            test_path = TRAINING_DATA_DIR / diff / "triplets_test.json"
            if test_path.exists():
                print(f"   ✅ Found test data in: {diff}")
                difficulty = diff
                break
        else:
            raise FileNotFoundError(f"No test data found in any difficulty level under {TRAINING_DATA_DIR}")
    
    with open(test_path, 'r', encoding='utf-8') as f:
        triplets = json.load(f)
    
    print(f"📊 Loaded {len(triplets)} test triplets from {difficulty} difficulty")
    return triplets, difficulty


def load_difficulty_stats(difficulty: str) -> Dict:
    """Load statistics metadata for a given difficulty if available."""
    stats_path = TRAINING_DATA_DIR / difficulty / "statistics.json"
    if stats_path.exists():
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Unable to parse statistics for %s", difficulty)
    return {}


def evaluate_model(model_path: str, triplets: List[Dict], difficulty: str, stats_info: Dict) -> Dict:
    """
    Evaluate an embedding model on test triplets.
    
    Returns:
        dict: Evaluation metrics including accuracy, mean similarities, and margin
    """
    print(f"\n{'='*80}")
    print(f"📊 Evaluating: {Path(model_path).name} ({difficulty} difficulty)")
    print(f"{'='*80}")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print("Loading model...")
    model = SentenceTransformer(model_path, device=device)
    
    # Extract texts
    anchors = [t['anchor'] for t in triplets]
    positives = [t['positive'] for t in triplets]
    negatives = [t['negative'] for t in triplets]
    
    # Generate embeddings with larger batch size for GPU efficiency
    batch_size = 32 if device == "cuda" else 8
    print(f"Generating embeddings for {len(triplets)} triplets (batch_size={batch_size})...")
    anchor_embs = model.encode(anchors, convert_to_tensor=True, show_progress_bar=True, 
                               batch_size=batch_size, device=device)
    positive_embs = model.encode(positives, convert_to_tensor=True, show_progress_bar=True, 
                                 batch_size=batch_size, device=device)
    negative_embs = model.encode(negatives, convert_to_tensor=True, show_progress_bar=True, 
                                 batch_size=batch_size, device=device)
    
    # Convert to numpy for similarity calculation
    anchor_embs = anchor_embs.cpu().numpy()
    positive_embs = positive_embs.cpu().numpy()
    negative_embs = negative_embs.cpu().numpy()
    
    # Calculate cosine similarities
    pos_similarities = np.array([
        cosine_similarity([anchor_embs[i]], [positive_embs[i]])[0][0]
        for i in range(len(triplets))
    ])
    
    neg_similarities = np.array([
        cosine_similarity([anchor_embs[i]], [negative_embs[i]])[0][0]
        for i in range(len(triplets))
    ])
    
    # Calculate metrics
    accuracy = np.mean(pos_similarities > neg_similarities)
    mean_pos_sim = np.mean(pos_similarities)
    mean_neg_sim = np.mean(neg_similarities)
    margin = np.mean(pos_similarities - neg_similarities)

    # Persist metrics to the structured logger for later review
    logger.info(
        f"Evaluation metrics for {Path(model_path).name} [{difficulty}]",
        extra={
            "Test Step": "evaluation",
            "Result": (
                f"accuracy={accuracy*100:.2f}%, "
                f"pos_sim={mean_pos_sim:.4f}, "
                f"neg_sim={mean_neg_sim:.4f}, "
                f"margin={margin:.4f}, "
                f"removed={stats_info.get('triplets_removed', 0)}"
            )
        },
    )

    summary_text = (
        "Evaluation summary:\n"
        "\n"
        f"Difficulty: {difficulty}\n"
        f"Accuracy: {accuracy*100:.2f}% — percentage of triplets where the positive outranks the negative.\n"
        f"Mean Positive similarity: {mean_pos_sim:.4f}\n"
        f"Mean Negative similarity: {mean_neg_sim:.4f}\n"
        f"Margin (positive - negative): {margin:.4f}\n"
        f"Triplets removed before export: {stats_info.get('triplets_removed', 0)} of {stats_info.get('triplets_found', len(triplets) + stats_info.get('triplets_removed', 0))}."
    )

    logger.info(summary_text)

    removed = int(stats_info.get("triplets_removed", 0))
    found = int(stats_info.get("triplets_found", len(triplets) + removed))
    
    return {
        'model': Path(model_path).name,
        'difficulty': difficulty,
        'accuracy': accuracy,
        'mean_pos_sim': mean_pos_sim,
        'mean_neg_sim': mean_neg_sim,
        'margin': margin,
        'triplets_removed': removed,
        'triplets_found': found,
    }


def print_metrics_table(results: List[Dict]) -> None:
    """Render evaluation metrics in a single table grouped by difficulty."""
    if not results:
        return

    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    measurement_specs = [
        ("Accuracy", lambda r: f"{r['accuracy']*100:.2f}%"),
        ("Mean Positive Similarity", lambda r: f"{r['mean_pos_sim']:.4f}"),
        ("Mean Negative Similarity", lambda r: f"{r['mean_neg_sim']:.4f}"),
        ("Margin (pos - neg)", lambda r: f"{r['margin']:.4f}"),
        (
            "Triplets Removed",
            lambda r: f"{r['triplets_removed']} of {r['triplets_found']}",
        ),
    ]

    ideal_ranges = {
        "easy": {
            "Accuracy": "95–100%",
            "Mean Positive Similarity": "0.75–0.95",
            "Mean Negative Similarity": "-0.10 to 0.20",
            "Margin (pos - neg)": "0.60–1.00",
            "Triplets Removed": "<10 of 243",
        },
        "medium": {
            "Accuracy": "80–95%",
            "Mean Positive Similarity": "0.70–0.90",
            "Mean Negative Similarity": "0.20–0.55",
            "Margin (pos - neg)": "0.25–0.60",
            "Triplets Removed": "<80 of 243",
        },
        "hard": {
            "Accuracy": "65–90%",
            "Mean Positive Similarity": "0.75–0.95",
            "Mean Negative Similarity": "0.45–0.80",
            "Margin (pos - neg)": "0.10–0.40",
            "Triplets Removed": "<110 of 243",
        },
    }

    normalized_results: List[Dict] = []
    for result in results:
        raw_diff = (result.get('difficulty') or '').strip().lower()
        normalized = {**result, 'difficulty': raw_diff or 'unknown'}
        normalized_results.append(normalized)

    sorted_results = sorted(
        normalized_results,
        key=lambda r: (difficulty_order.get(r['difficulty'], 99), r['difficulty'])
    )

    deduped_results: List[Dict] = []
    seen_difficulties = set()
    for result in sorted_results:
        diff = result['difficulty']
        if diff in seen_difficulties:
            continue
        seen_difficulties.add(diff)
        deduped_results.append(result)

    header = f"{'Measurement':<26} {'Difficulty':<12} {'Value':<18} Ideal range"
    print(f"\n{'='*80}")
    print(f"📊 Evaluation Metrics — {deduped_results[0]['model']}")
    print(f"{'='*80}")
    print(header)
    print(f"{'-'*80}")

    for idx, result in enumerate(deduped_results):
        difficulty = result['difficulty'].capitalize()
        for measurement, formatter in measurement_specs:
            ideal = ideal_ranges.get(result['difficulty'], {}).get(measurement, "—")
            print(f"{measurement:<26} {difficulty:<12} {formatter(result):<18} {ideal}")
        if idx < len(deduped_results) - 1:
            print(f"{'-'*80}")

    print(f"{'='*80}\n")


def compare_models(results):
    """Print a comparison table of multiple model results."""
    print(f"\n{'='*80}")
    print(f"📊 MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"{'Model':<40} {'Accuracy':>10} {'Margin':>10}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['model']:<40} {r['accuracy']*100:>9.2f}% {r['margin']:>10.4f}")
    
    print(f"{'='*80}\n")
    
    # Find best model
    best = max(results, key=lambda x: x['accuracy'])
    print(f"🏆 Best model: {best['model']} ({best['accuracy']*100:.2f}% accuracy)")


def main():
    """CLI entry point that wires up args and triggers evaluations."""
    parser = argparse.ArgumentParser(description="Evaluate trained embedding models")
    parser.add_argument("--epoch", type=str, default="3", 
                       help="Which epoch to test: 1, 2, 3, or 'base' (default: 3)")
    parser.add_argument("--all", action="store_true",
                       help="Test all models (base, epoch1, epoch2, epoch3) and compare")
    parser.add_argument("--difficulty", type=str, default="all",
                       choices=["easy", "medium", "hard", "all"],
                       help="Which difficulty test set to use (default: all)")
    
    args = parser.parse_args()
    
    difficulty_targets = ["easy", "medium", "hard"] if args.difficulty == "all" else [args.difficulty]
    
    if args.all:
        # Evaluate all models
        compare_inputs = []
        for epoch in ["base", 1, 2, 3]:
            try:
                model_path = get_model_path(epoch)
                if not Path(model_path).exists():
                    print(f"⚠️  Skipping {epoch} - model not found at {model_path}")
                    continue
                per_model_results = []
                for difficulty in difficulty_targets:
                    try:
                        triplets, resolved_diff = load_test_data(difficulty)
                    except FileNotFoundError as e:
                        print(f"❌ {e}")
                        continue
                    stats_info = load_difficulty_stats(resolved_diff)
                    result = evaluate_model(model_path, triplets, resolved_diff, stats_info)
                    per_model_results.append(result)
                if per_model_results:
                    print_metrics_table(per_model_results)
                    compare_inputs.append({
                        "model": Path(model_path).name,
                        "accuracy": float(np.mean([r['accuracy'] for r in per_model_results])),
                        "margin": float(np.mean([r['margin'] for r in per_model_results])),
                    })
            except Exception as e:
                print(f"❌ Error evaluating epoch {epoch}: {e}")
        
        if len(compare_inputs) > 1:
            compare_models(compare_inputs)
    else:
        # Evaluate single model
        epoch = args.epoch if args.epoch == "base" else int(args.epoch)
        model_path = get_model_path(epoch)
        
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return
        
        per_model_results = []
        for difficulty in difficulty_targets:
            triplets, resolved_diff = load_test_data(difficulty)
            stats_info = load_difficulty_stats(resolved_diff)
            result = evaluate_model(model_path, triplets, resolved_diff, stats_info)
            per_model_results.append(result)

        if per_model_results:
            print_metrics_table(per_model_results)


if __name__ == "__main__":
    main()
