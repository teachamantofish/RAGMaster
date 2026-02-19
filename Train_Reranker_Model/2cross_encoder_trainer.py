"""Cross-encoder trainer for the reranker stage."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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


def _load_examples(split: str, difficulties: Iterable[str], limit: int | None = None,
                    target_pos_neg_ratio: float = 0.33) -> List[InputExample]:
    """Load cross-encoder pairs with optional positive upsampling.

    When *target_pos_neg_ratio* > 0 the positives are duplicated so that
    ``n_pos / n_neg ≈ target_pos_neg_ratio`` (default ~1:3).  Set to 0 to
    disable upsampling.
    """
    positives: List[InputExample] = []
    negatives: List[InputExample] = []
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
                ex = InputExample(texts=[record["query"], record["candidate"]], label=label)
                if label >= 0.5:
                    positives.append(ex)
                else:
                    negatives.append(ex)
                used += 1
                if limit and used >= limit:
                    break
        if limit and used >= limit:
            break

    # --- positive upsampling to fix class imbalance ---
    n_pos, n_neg = len(positives), len(negatives)
    logger.info("Loaded %d positives, %d negatives (ratio 1:%.1f) for %s split",
                n_pos, n_neg, n_neg / max(n_pos, 1), split)

    if target_pos_neg_ratio > 0 and n_pos > 0 and n_neg > 0:
        desired_pos = int(n_neg * target_pos_neg_ratio)
        if desired_pos > n_pos:
            repeats = desired_pos // n_pos
            remainder = desired_pos % n_pos
            upsampled = positives * repeats + positives[:remainder]
            logger.info("Upsampled positives: %d → %d (target ratio 1:%.0f)",
                        n_pos, len(upsampled), 1.0 / target_pos_neg_ratio)
            positives = upsampled

    examples = positives + negatives
    return examples


def _build_dataloader(examples: List[InputExample], batch_size: int, shuffle: bool = True) -> DataLoader:
    if not examples:
        raise ValueError("No examples provided to the dataloader")
    return DataLoader(examples, shuffle=shuffle, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Grouped data loading for listwise ranking loss
# ---------------------------------------------------------------------------

def _load_grouped(split: str, difficulties: Iterable[str],
                  limit: int | None = None) -> Dict[str, Dict]:
    """Load pairs grouped by query_id.

    Returns ``{query_id: {"query": str, "positives": [(cand, rank)], "negatives": [(cand, rank)]}}``.
    """
    groups: Dict[str, Dict] = {}
    used = 0
    for difficulty in difficulties:
        path = CROSS_ENCODER_DATA_DIR / difficulty / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                qid = rec["query_id"]
                if qid not in groups:
                    groups[qid] = {"query": rec["query"], "positives": [], "negatives": []}
                entry = (rec["candidate"], rec.get("rank", 999))
                if float(rec.get("label", 0)) >= 0.5:
                    groups[qid]["positives"].append(entry)
                else:
                    groups[qid]["negatives"].append(entry)
                used += 1
                if limit and used >= limit:
                    break
        if limit and used >= limit:
            break

    logger.info("Loaded %d query groups for %s split (%d total pairs)",
                len(groups), split, used)
    for qid in sorted(groups):
        g = groups[qid]
        logger.info("  %s: %d pos, %d neg", qid, len(g["positives"]), len(g["negatives"]))
    return groups


def _train_listwise(model: CrossEncoder, groups: Dict[str, Dict],
                    val_examples: List[InputExample] | None,
                    epochs: int, warmup_steps: int, learning_rate: float,
                    use_amp: bool, max_length: int,
                    neg_per_positive: int = 15,
                    temperature: float = 1.0,
                    accumulation_steps: int = 1) -> None:
    """Custom listwise training loop with InfoNCE / grouped softmax loss.

    For each training step we sample one query, pick one of its positives,
    and sample *neg_per_positive* hard negatives.  The model scores all
    candidates; the loss pushes the positive score above all negatives via
    softmax cross-entropy (positive is always index 0).

    Anti-overfitting measures (Exp 12):
    - temperature scaling (τ) sharpens/softens the softmax
    - gradient accumulation across multiple groups per optimizer step
    - cosine annealing schedule (warmup then decay to 0)
    """
    device = model.model.device
    tokenizer = model.tokenizer
    hf_model = model.model
    hf_model.train()

    # Ensure pad token is set (Qwen models may not have one configured)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hf_model.config.pad_token_id is None:
        hf_model.config.pad_token_id = tokenizer.pad_token_id

    # Build flat list of (query_id, pos_idx) covering every positive once per epoch
    all_items: List[Tuple[str, int]] = []
    for qid, g in groups.items():
        for pidx in range(len(g["positives"])):
            all_items.append((qid, pidx))

    steps_per_epoch = len(all_items)
    total_steps = steps_per_epoch * epochs
    optimizer_steps = total_steps // accumulation_steps
    logger.info("Listwise training: %d queries, %d positives/epoch, %d total fwd steps, "
                "%d optimizer steps, neg_per_positive=%d, group_size=%d, "
                "temperature=%.3f, accumulation=%d",
                len(groups), steps_per_epoch, total_steps,
                optimizer_steps, neg_per_positive,
                neg_per_positive + 1, temperature, accumulation_steps)

    optimizer = torch.optim.AdamW(hf_model.parameters(), lr=learning_rate)

    # Cosine annealing with linear warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(optimizer_steps - warmup_steps, 1)
        return 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    evaluator = SimpleCEBinaryEvaluator(val_examples, name="reranker_val") if val_examples else None
    global_step = 0
    optimizer_step_count = 0

    for epoch in range(1, epochs + 1):
        random.shuffle(all_items)
        epoch_loss = 0.0
        accum_loss = 0.0

        optimizer.zero_grad()

        for step_in_epoch, (qid, pidx) in enumerate(all_items, 1):
            g = groups[qid]
            query_text = g["query"]
            pos_text = g["positives"][pidx][0]

            # Sample hard negatives (prefer top-ranked, i.e. hardest)
            negs = g["negatives"]
            if neg_per_positive > 0 and len(negs) > neg_per_positive:
                neg_sample = random.sample(negs[:min(len(negs), neg_per_positive * 3)], neg_per_positive)
            else:
                neg_sample = negs  # use ALL negatives

            # Build pairs: positive at index 0, then negatives
            pairs = [(query_text, pos_text)] + [(query_text, n[0]) for n in neg_sample]

            # Tokenize
            texts_a = [p[0] for p in pairs]
            texts_b = [p[1] for p in pairs]
            encoded = tokenizer(
                texts_a, texts_b,
                padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            # Target: positive is always index 0
            target = torch.tensor([0], dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = hf_model(**encoded)
                logits = outputs.logits.squeeze(-1)  # (group_size,)
                # InfoNCE: softmax CE over the group, target=0 (positive)
                # Temperature scaling sharpens gradients for hard negatives
                scaled_logits = logits.unsqueeze(0) / temperature
                loss = F.cross_entropy(scaled_logits, target)
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item() * accumulation_steps  # unscaled for logging

            # Optimizer step every accumulation_steps forward passes
            if step_in_epoch % accumulation_steps == 0 or step_in_epoch == steps_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(hf_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                optimizer_step_count += 1

            global_step += 1
            epoch_loss += accum_loss if (step_in_epoch % accumulation_steps == 0 or step_in_epoch == steps_per_epoch) else 0

            if global_step % 25 == 0:
                avg = epoch_loss / max(step_in_epoch // accumulation_steps, 1)
                lr_now = scheduler.get_last_lr()[0]
                logger.info("epoch %d step %d/%d  loss=%.4f  avg_loss=%.4f  lr=%.2e  group_size=%d",
                            epoch, step_in_epoch, steps_per_epoch,
                            accum_loss, avg, lr_now, len(pairs))

            if step_in_epoch % accumulation_steps == 0:
                accum_loss = 0.0

        actual_opt_steps = (steps_per_epoch + accumulation_steps - 1) // accumulation_steps
        avg_epoch_loss = epoch_loss / max(actual_opt_steps, 1)
        logger.info("Epoch %d complete — avg_loss=%.4f (%d optimizer steps)", epoch, avg_epoch_loss, actual_opt_steps)

        if evaluator:
            score = evaluator(model, epoch=epoch, steps=global_step)
            logger.info("Epoch %d val accuracy: %.4f", epoch, score)

    logger.info("Listwise training complete — %d total steps", global_step)


def _train(args):
    difficulties = [d.lower() for d in args.difficulties]
    loss_type = getattr(args, "loss", "binary_ce")

    if loss_type == "listwise":
        # --- Grouped softmax (InfoNCE) training ---
        # Listwise only works with 'hard' difficulty (retrieval-based pairs with query_id).
        # Easy/medium are legacy triplet data without query grouping.
        listwise_diffs = ["hard"]
        groups = _load_grouped("train", listwise_diffs, args.max_train)
        if not groups:
            raise SystemExit("No training groups found. Run 1create_training_data.py first.")

        val_limit = args.max_eval or 2000
        val_examples = _load_examples("test", difficulties, val_limit, target_pos_neg_ratio=0)

        device = _resolve_device(args.device)
        max_length = args.max_length or RERANKER_TRAINING_CONFIG.get("max_length", 256)
        learning_rate = args.learning_rate or RERANKER_TRAINING_CONFIG.get("learning_rate", 2e-5)
        use_amp = RERANKER_TRAINING_CONFIG.get("use_amp", True) if args.use_amp is None else args.use_amp
        epochs = RERANKER_TRAINING_CONFIG.get("epochs", 3)
        warmup_steps = RERANKER_TRAINING_CONFIG.get("warmup_steps", 100)
        neg_per_positive = RERANKER_TRAINING_CONFIG.get("neg_per_positive", 15)
        temperature = RERANKER_TRAINING_CONFIG.get("temperature", 1.0)
        accumulation_steps = RERANKER_TRAINING_CONFIG.get("accumulation_steps", 1)

        logger.info("Listwise training on device %s (lr=%s max_len=%s amp=%s neg_per_pos=%d temp=%.3f accum=%d)",
                     device, learning_rate, max_length, use_amp, neg_per_positive, temperature, accumulation_steps)

        model = CrossEncoder(CONFIG_MODEL_NAME, num_labels=1, max_length=max_length, device=device)
        RERANKER_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        _train_listwise(
            model, groups, val_examples,
            epochs=epochs, warmup_steps=warmup_steps,
            learning_rate=learning_rate, use_amp=use_amp,
            max_length=max_length, neg_per_positive=neg_per_positive,
            temperature=temperature, accumulation_steps=accumulation_steps,
        )

        model.save(str(RERANKER_OUTPUT_PATH))
        logger.info("Reranker saved to %s", RERANKER_OUTPUT_PATH)
        return

    # --- Original binary CE training ---
    train_examples = _load_examples("train", difficulties, args.max_train, target_pos_neg_ratio=0.33)
    # Cap val set to 2000 examples — full evaluation happens in the separate evaluate stage.
    # The 37K+ raw val pairs cause OOM/stalls during model.predict() at epoch end.
    val_limit = args.max_eval or 2000
    val_examples = _load_examples("test", difficulties, val_limit, target_pos_neg_ratio=0)

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

    # --- Optional: freeze base model for linear-probe training ---
    freeze_base = RERANKER_TRAINING_CONFIG.get("freeze_base", False)
    if freeze_base:
        frozen_count = 0
        trainable_count = 0
        for name, param in model.model.named_parameters():
            if "score" in name:
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()
        logger.info("Freeze-base mode: %d params frozen, %d trainable (score head only)",
                    frozen_count, trainable_count)

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
    parser.add_argument(
        "--loss",
        choices=["binary_ce", "listwise"],
        default="binary_ce",
        help="Training loss: binary_ce (original CrossEncoder.fit) or listwise (grouped softmax/InfoNCE)",
    )

    args = parser.parse_args()

    if args.action == "train":
        _train(args)
    else:
        _evaluate(args)


if __name__ == "__main__":
    main()
