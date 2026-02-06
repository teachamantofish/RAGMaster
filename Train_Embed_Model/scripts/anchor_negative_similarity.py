"""Quick utility to measure anchor-vs-negative cosine similarity for triplet files."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Ensure the Train_Embed_Model root is on sys.path so we can import sibling modules
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.config_training_embed import CONFIG_MODEL_NAME, TRAINING_DATA_DIR, LOG_FILES
from scripts.custom_logger import setup_global_logger

script_base = Path(__file__).stem
LOG_HEADER = ["Date", "Level", "Message", "Test Step", "Result"]
logger = setup_global_logger(script_name=script_base, cwd=LOG_FILES, log_level="INFO", headers=LOG_HEADER)


def _default_triplet_path(difficulty: str, split: str) -> Path:
    candidate = TRAINING_DATA_DIR / difficulty / f"triplets_{split}.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No triplet file found at {candidate}. Supply --triplet-file explicitly.")


def _load_triplets(path: Path, limit: int | None) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of triplets in {path}, got {type(data)}")
    if limit is not None:
        data = data[:limit]
    for idx, triplet in enumerate(data):
        if "anchor" not in triplet or "negative" not in triplet:
            raise KeyError(f"Triplet {idx} missing required fields in {path}")
    return data


def _batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def _encode_pairs(model: SentenceTransformer, anchors: List[str], negatives: List[str], batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    anchor_embs: List[np.ndarray] = []
    negative_embs: List[np.ndarray] = []
    for a_batch, n_batch in zip(_batched(anchors, batch_size), _batched(negatives, batch_size)):
        anchor_embs.append(
            model.encode(
                a_batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=len(a_batch),
                show_progress_bar=False,
            )
        )
        negative_embs.append(
            model.encode(
                n_batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=len(n_batch),
                show_progress_bar=False,
            )
        )
    return np.vstack(anchor_embs), np.vstack(negative_embs)


def _summarize(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def _write_details(path: Path, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "anchor_negative_cosine"])
        for idx, value in enumerate(scores.tolist()):
            writer.writerow([idx, value])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute anchor-negative cosine similarity for triplet files.")
    parser.add_argument("--triplet-file", type=Path, default=None, help="Path to triplets JSON. Defaults to TRAINING_DATA_DIR/<difficulty>/triplets_<split>.json")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="hard", help="Difficulty subfolder to use when --triplet-file is omitted.")
    parser.add_argument("--split", choices=["train", "test"], default="train", help="Dataset split to use when --triplet-file is omitted.")
    parser.add_argument("--model-path", type=Path, default=Path(CONFIG_MODEL_NAME), help="Embedding model to load (defaults to CONFIG_MODEL_NAME).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of triplets to evaluate.")
    parser.add_argument("--details-csv", type=Path, default=None, help="If provided, write per-triplet scores to this CSV path.")
    args = parser.parse_args()

    try:
        triplet_path = args.triplet_file if args.triplet_file else _default_triplet_path(args.difficulty, args.split)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not triplet_path.exists():
        parser.error(f"Triplet file not found: {triplet_path}")

    triplets = _load_triplets(triplet_path, args.limit)
    anchors = [t["anchor"] for t in triplets]
    negatives = [t["negative"] for t in triplets]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model_path} on {device}...")
    logger.info(f"Loading model from {args.model_path}")
    model = SentenceTransformer(str(args.model_path), device=device)

    print(f"Encoding {len(triplets)} triplets...")
    anchor_embs, negative_embs = _encode_pairs(model, anchors, negatives, args.batch_size)
    scores = (anchor_embs * negative_embs).sum(axis=1)
    summary = _summarize(scores)

    summary_text = (
        f"Triplet file: {triplet_path}\n"
        f"Count: {summary['count']}\n"
        f"Mean cosine(anchor, negative): {summary['mean']:.4f}\n"
        f"Median: {summary['median']:.4f}\n"
        f"Min/Max: {summary['min']:.4f} / {summary['max']:.4f}\n"
        f"Std Dev: {summary['std']:.4f}"
    )
    print(summary_text)
    logger.info(summary_text, extra={"Test Step": "cosine_stats", "Result": f"mean={summary['mean']:.4f}"})

    if args.details_csv:
        _write_details(args.details_csv, scores)
        print(f"Per-triplet scores written to {args.details_csv}")
        logger.info(f"Per-triplet scores saved", extra={"Test Step": "write_csv", "Result": str(args.details_csv)})


if __name__ == "__main__":
    main()
