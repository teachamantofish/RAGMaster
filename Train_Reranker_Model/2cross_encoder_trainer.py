"""Cross-encoder trainer for the reranker stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

from scripts.config_training_rerank import (
    CROSS_ENCODER_DATA_DIR,
    RERANKER_TRAINING_CONFIG,
    CONFIG_MODEL_NAME,
    RERANKER_OUTPUT_PATH,
    RERANK_EVAL_CONFIG,
)
from scripts.custom_logger import setup_global_logger

logger = setup_global_logger(
    script_name="4cross_encoder_trainer",
    cwd=CROSS_ENCODER_DATA_DIR,
    log_level="INFO",
    headers=["Date", "Level", "Message", "Split", "Pairs"],
)


class SimpleCEBinaryEvaluator:
    """Minimal evaluator compatible with CrossEncoder.fit."""

    def __init__(self, examples: List[InputExample], name: str = "reranker_eval", threshold: float = 0.5):
        self.examples = examples
        self.name = name
        self.threshold = threshold

    def __call__(self, model: CrossEncoder, epoch: int = 0, steps: int = 0, **_: object) -> float:
        if not self.examples:
            return 0.0
        sentences = [[ex.texts[0], ex.texts[1]] for ex in self.examples]
        labels = np.array([ex.label for ex in self.examples], dtype=np.float32)
        scores = np.array(model.predict(sentences), dtype=np.float32)
        preds = (scores >= self.threshold).astype(np.float32)
        accuracy = float((preds == labels).mean())
        logger.info("Evaluator %s — epoch %s step %s accuracy %.4f", self.name, epoch, steps, accuracy)
        return accuracy


def _resolve_device(requested: str | None = None) -> str:
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise SystemExit("CUDA requested but torch.cuda.is_available() returned False")
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_directml  # noqa: F401
        return "dml"
    except Exception:
        return "cpu"


def _load_examples(split: str, difficulties: Iterable[str], limit: int | None = None) -> List[InputExample]:
    examples: List[InputExample] = []
    used = 0

    for difficulty in difficulties:
        path = CROSS_ENCODER_DATA_DIR / difficulty / f"{split}.jsonl"
        if not path.exists():
            logger.warning("No %s data found for %s split", difficulty, split)
            continue

        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                label = float(record.get("label", 0))
                examples.append(InputExample(texts=[record["query"], record["candidate"]], label=label))
                used += 1
                if limit and used >= limit:
                    return examples

    return examples


def _build_dataloader(examples: List[InputExample], batch_size: int, shuffle: bool = True) -> DataLoader:
    if not examples:
        raise ValueError("No examples provided to the dataloader")
    return DataLoader(examples, shuffle=shuffle, batch_size=batch_size)


def _train(args):
    difficulties = [d.lower() for d in args.difficulties]
    train_examples = _load_examples("train", difficulties, args.max_train)
    val_examples = _load_examples("test", difficulties, args.max_eval)

    if not train_examples:
        raise SystemExit("No training pairs found. Run 1create_training_data.py first.")

    device = _resolve_device(args.device)
    batch_size = args.batch_size or RERANKER_TRAINING_CONFIG.get("batch_size", 16)
    max_length = args.max_length or RERANKER_TRAINING_CONFIG.get("max_length", 512)
    learning_rate = args.learning_rate or RERANKER_TRAINING_CONFIG.get("learning_rate", 2e-5)
    use_amp = (
        RERANKER_TRAINING_CONFIG.get("use_amp", True)
        if args.use_amp is None
        else args.use_amp
    )

    logger.info(
        "Training with %d pairs across %d difficulties on device %s (batch=%s lr=%s max_len=%s amp=%s)",
        len(train_examples),
        len(difficulties),
        device,
        batch_size,
        learning_rate,
        max_length,
        use_amp,
    )
    model = CrossEncoder(
        CONFIG_MODEL_NAME,
        num_labels=1,
        max_length=max_length,
        device=device,
    )

    train_dataloader = _build_dataloader(train_examples, batch_size, shuffle=True)
    evaluator = SimpleCEBinaryEvaluator(val_examples, name="reranker_val") if val_examples else None

    RERANKER_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=RERANKER_TRAINING_CONFIG.get("epochs", 3),
        warmup_steps=RERANKER_TRAINING_CONFIG.get("warmup_steps", 100),
        output_path=str(RERANKER_OUTPUT_PATH),
        optimizer_params={"lr": learning_rate},
        use_amp=use_amp,
    )

    # Explicit save guarantees downstream evaluation finds config/tokenizer artifacts.
    model.save(str(RERANKER_OUTPUT_PATH))

    logger.info("Reranker saved to %s", RERANKER_OUTPUT_PATH)


def _evaluate(args):
    difficulties = [d.lower() for d in args.difficulties]
    split = args.split
    examples = _load_examples(split, difficulties, args.max_eval)
    if not examples:
        raise SystemExit(f"No {split} data available for difficulties: {difficulties}")

    model_path = Path(args.model_path or RERANKER_OUTPUT_PATH)
    if not model_path.exists():
        raise SystemExit(f"Model not found at {model_path}")

    device = _resolve_device(args.device)
    logger.info("Evaluating %s on %d pairs using device %s", model_path, len(examples), device)
    evaluator = SimpleCEBinaryEvaluator(examples, name=f"reranker_{split}")
    model = CrossEncoder(
        str(model_path),
        num_labels=1,
        max_length=RERANKER_TRAINING_CONFIG.get("max_length", 512),
        device=device,
    )
    score = evaluator(model)
    logger.info("Evaluation accuracy (%s): %.4f", split, score)


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate the reranker cross-encoder")
    parser.add_argument("--action", choices=["train", "evaluate"], default="train")
    parser.add_argument("--difficulties", nargs="*", default=RERANK_EVAL_CONFIG["difficulties"], help="Difficulty buckets to use")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Split to evaluate when --action evaluate")
    parser.add_argument("--model-path", default=str(RERANKER_OUTPUT_PATH), help="Model checkpoint to load for evaluation")
    parser.add_argument("--max-train", type=int, default=None, help="Optional cap on training pairs for smoke tests")
    parser.add_argument("--max-eval", type=int, default=None, help="Optional cap on evaluation pairs")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device. 'auto' prefers CUDA when available",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override optimizer learning rate")
    parser.add_argument("--max-length", type=int, default=None, help="Override max token length")
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force enable/disable mixed precision (default uses config)",
    )

    args = parser.parse_args()

    if args.action == "train":
        _train(args)
    else:
        _evaluate(args)


if __name__ == "__main__":
    main()
