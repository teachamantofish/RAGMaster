"""Minimal orchestration script for the reranker workflow.

Stages:
- data: run build_retrieval_candidates.py (or legacy triplet builder) to refresh
    cross-encoder pairs directly from the production retriever
- train: run 2cross_encoder_trainer.py with optional overrides
- evaluate: run 3evaluate_model.py to compare reranker vs retriever baseline
- full: execute data → train → evaluate
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.config_training_rerank import (
    BASE_CWD,
    CODE_CHANGE,
    CROSS_ENCODER_DATA_DIR,
    LOG_FILES,
    MASTER_RERANK_LOG,
    RERANK_EVAL_CONFIG,
    RERANKER_TRAINING_CONFIG,
    RETRIEVER_PIPELINE_CONFIG,
)
from scripts.custom_logger import setup_global_logger


PROJECT_ROOT = Path(__file__).resolve().parent


logger = setup_global_logger(
    script_name="0pipeline_manager",
    cwd=CROSS_ENCODER_DATA_DIR,
    log_level="INFO",
    headers=["Date", "Level", "Message", "Stage", "Command"],
)

EVAL_SUMMARY_FILENAME = "reranker_eval_summary.json"
EVAL_OUTPUT_DIR_DEFAULT = CROSS_ENCODER_DATA_DIR / "eval_outputs"
CSV_FIELDS = [
    "timestamp",
    "BASE_CWD",
    "CODE_CHANGE",
    "max_length",
    "batch_size",
    "lr",
    "epochs",
    "candidate_k",
    "num_chunks",
    "num_eval_queries",
    "baseline_mrr",
    "rerank_mrr",
    "baseline_recall5",
    "rerank_recall5",
    "baseline_ndcg10",
    "rerank_ndcg10",
]
CSV_FIELDS.extend(["retrieval_seconds", "rerank_seconds", "total_seconds"])


def _python_executable() -> str:
    venv_python = Path(".venv") / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _run(cmd: List[str], description: str, dry_run: bool) -> Tuple[bool, float]:
    logger.info("Stage %s", description, extra={"Stage": description, "Command": " ".join(cmd)})
    start = time.perf_counter()
    if dry_run:
        print(f"[DRY RUN] {description}: {' '.join(cmd)}")
        return True, 0.0
    print(f"\n{'=' * 80}\n🚀 {description}\n{'=' * 80}")
    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        print(f"✅ {description} completed successfully")
        return True, time.perf_counter() - start
    except subprocess.CalledProcessError as exc:
        print(f"❌ {description} failed (exit code {exc.returncode})")
        return False, time.perf_counter() - start


def _run_data_stage(
    python_exe: str,
    args,
    dry_run: bool,
    force_validate_only: bool = False,
    *,
    stats_output: Path | None = None,
) -> Tuple[bool, float]:
    if args.use_legacy_triplets:
        cmd = [python_exe, "1create_training_data.py"]
        if args.data_regenerate_triplets:
            cmd.append("--regenerate-triplets")
            if args.data_chunks_file:
                cmd.extend(["--chunks-file", str(args.data_chunks_file)])
        description = "Generate reranker data (legacy triplets)"
    else:
        cmd = [python_exe, "build_retrieval_candidates.py"]
        if args.data_queries_file:
            cmd.extend(["--queries-file", str(args.data_queries_file)])
        if args.data_top_k:
            cmd.extend(["--top-k", str(args.data_top_k)])
        if args.data_max_queries:
            cmd.extend(["--max-queries", str(args.data_max_queries)])
        if args.data_lexical_weight is not None:
            cmd.extend(["--lexical-weight", str(args.data_lexical_weight)])
        if args.data_dense_weight is not None:
            cmd.extend(["--dense-weight", str(args.data_dense_weight)])
        if args.data_batch_size:
            cmd.extend(["--batch-size", str(args.data_batch_size)])
        if args.data_validate_only or force_validate_only:
            cmd.append("--validate-only")
        if args.data_allow_missed_positives:
            cmd.append("--no-fail-on-missed-positives")
        if args.data_allow_missing_ground_truth:
            cmd.append("--no-fail-on-missing-ground-truth")
        if stats_output:
            cmd.extend(["--stats-output", str(stats_output)])
        description = "Build retrieval candidates"
    return _run(cmd, description, dry_run)


def _run_train_stage(python_exe: str, args, dry_run: bool) -> Tuple[bool, float]:
    cmd = [python_exe, "2cross_encoder_trainer.py", "--action", "train", "--device", args.device]
    if args.difficulties:
        cmd.extend(["--difficulties", *args.difficulties])
    if args.max_train:
        cmd.extend(["--max-train", str(args.max_train)])
    if args.max_eval:
        cmd.extend(["--max-eval", str(args.max_eval)])
    if args.trainer_batch_size:
        cmd.extend(["--batch-size", str(args.trainer_batch_size)])
    if args.trainer_learning_rate:
        cmd.extend(["--learning-rate", str(args.trainer_learning_rate)])
    if args.trainer_max_length:
        cmd.extend(["--max-length", str(args.trainer_max_length)])
    if args.use_amp is not None:
        cmd.append("--use-amp" if args.use_amp else "--no-use-amp")
    return _run(cmd, "Train reranker cross-encoder", dry_run)


def _run_eval_stage(python_exe: str, args, dry_run: bool, *, output_dir: Path) -> Tuple[bool, float]:
    cmd = [python_exe, "3evaluate_model.py", "--split", args.split]
    if args.difficulties:
        cmd.extend(["--difficulties", *args.difficulties])
    if args.model_path:
        cmd.extend(["--model-path", args.model_path])
    if args.baseline_model:
        cmd.extend(["--baseline-model", args.baseline_model])
    if args.baseline_batch_size:
        cmd.extend(["--baseline-batch-size", str(args.baseline_batch_size)])
    if args.cross_batch_size:
        cmd.extend(["--cross-batch-size", str(args.cross_batch_size)])
    cmd.extend(["--output-dir", str(output_dir)])
    return _run(cmd, "Evaluate reranker vs retriever baseline", dry_run)


def _create_stats_path() -> Path:
    stamp = int(time.time() * 1000)
    return LOG_FILES / f".retrieval_stats_{stamp}.json"


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _aggregate_metrics(summary: Dict[str, Dict]) -> Dict[str, Dict]:
    if not summary:
        return {}
    agg: Dict[str, Dict] = {
        "pairs": 0,
        "anchors": 0,
        "baseline": defaultdict(float),
        "reranker": defaultdict(float),
    }
    for stats in summary.values():
        agg["pairs"] += stats.get("pairs", 0)
        agg["anchors"] += stats.get("anchors", 0)
        for key, value in stats.get("baseline", {}).items():
            agg["baseline"][key] += value
        for key, value in stats.get("reranker", {}).items():
            agg["reranker"][key] += value
    count = len(summary)
    if count:
        agg["baseline"] = {k: v / count for k, v in agg["baseline"].items()}
        agg["reranker"] = {k: v / count for k, v in agg["reranker"].items()}
    return agg


def _resolved_training_params(args) -> Dict[str, float]:
    return {
        "batch_size": args.trainer_batch_size or RERANKER_TRAINING_CONFIG.get("batch_size"),
        "learning_rate": args.trainer_learning_rate or RERANKER_TRAINING_CONFIG.get("learning_rate"),
        "max_length": args.trainer_max_length or RERANKER_TRAINING_CONFIG.get("max_length"),
        "epochs": RERANKER_TRAINING_CONFIG.get("epochs"),
    }


def _resolve_candidate_k(args) -> int:
    return args.data_top_k or RETRIEVER_PIPELINE_CONFIG.get("top_k", 0)


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def _format_metric(value) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _format_seconds(value) -> str:
    if value is None:
        return ""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return ""
    rounded = round(seconds, 1)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.1f}"


def _format_learning_rate(value) -> str:
    if value is None:
        return ""
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return str(value)
    if rate == 0:
        return "0.000000"
    if abs(rate) < 1e-3:
        formatted = f"{rate:.0e}".replace("E", "e")
        return formatted.replace("e-0", "e-").replace("e+0", "e+")
    return f"{rate:.6f}"


def _append_master_log(row: Dict[str, str]) -> None:
    target = LOG_FILES / MASTER_RERANK_LOG
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists()
    with open(target, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _build_master_row(
    args,
    *,
    retrieval_stats: Dict[str, Any],
    eval_payload: Dict[str, Any],
    durations: Dict[str, float],
) -> Dict[str, str]:
    summary = eval_payload.get("summary", {})
    aggregate = eval_payload.get("aggregate") or _aggregate_metrics(summary)
    coverage = eval_payload.get("coverage") or {}
    overall_baseline = (aggregate or {}).get("baseline", {})
    overall_reranker = (aggregate or {}).get("reranker", {})

    training = _resolved_training_params(args)
    candidate_k = _resolve_candidate_k(args)
    num_eval_queries = (
        retrieval_stats.get("queries_total")
        if isinstance(retrieval_stats, dict)
        else None
    ) or coverage.get("queries_total") or (aggregate or {}).get("anchors")
    durations = durations or {}
    total_seconds = sum(durations.values())

    row: Dict[str, str] = {
        "timestamp": datetime.now().strftime("%m/%d/%y %H:%M"),
        "BASE_CWD": BASE_CWD.name or BASE_CWD.as_posix(),
        "CODE_CHANGE": CODE_CHANGE,
        "max_length": _format_value(training.get("max_length")),
        "batch_size": _format_value(training.get("batch_size")),
        "lr": _format_learning_rate(training.get("learning_rate")),
        "epochs": _format_value(training.get("epochs")),
        "candidate_k": _format_value(candidate_k),
        "num_chunks": _format_value(retrieval_stats.get("chunk_count")),
        "num_eval_queries": _format_value(num_eval_queries),
        "baseline_mrr": _format_metric(overall_baseline.get("mrr")),
        "rerank_mrr": _format_metric(overall_reranker.get("mrr")),
        "baseline_recall5": _format_metric(overall_baseline.get("recall@5")),
        "rerank_recall5": _format_metric(overall_reranker.get("recall@5")),
        "baseline_ndcg10": _format_metric(overall_baseline.get("ndcg@10")),
        "rerank_ndcg10": _format_metric(overall_reranker.get("ndcg@10")),
        "retrieval_seconds": _format_seconds(durations.get("data")),
        "rerank_seconds": _format_seconds(durations.get("train")),
        "total_seconds": _format_seconds(total_seconds),
    }

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Reranker pipeline manager")
    parser.add_argument(
        "--stage",
        choices=["validate", "data", "train", "evaluate", "full"],
        default="full",
        help="Which stage to run",
    )
    parser.add_argument(
        "--difficulties",
        nargs="*",
        default=RERANK_EVAL_CONFIG["difficulties"],
        help="Difficulty buckets for training/eval",
    )
    parser.add_argument(
        "--use-legacy-triplets",
        action="store_true",
        help="Use the synthetic triplet builder instead of the production retriever",
    )
    parser.add_argument(
        "--data-regenerate-triplets",
        action="store_true",
        help="Force a fresh triplet build before exporting cross-encoder pairs",
    )
    parser.add_argument(
        "--data-chunks-file",
        type=Path,
        help="Custom chunk JSON when --data-regenerate-triplets is used",
    )
    parser.add_argument(
        "--data-queries-file",
        type=Path,
        help="Override the labeled queries file for build_retrieval_candidates",
    )
    parser.add_argument(
        "--data-top-k",
        type=int,
        help="Override the retrieval top-k for build_retrieval_candidates",
    )
    parser.add_argument(
        "--data-max-queries",
        type=int,
        help="Optional cap on labeled queries processed during the data stage",
    )
    parser.add_argument(
        "--data-lexical-weight",
        type=float,
        help="Override lexical weight for hybrid retrieval",
    )
    parser.add_argument(
        "--data-dense-weight",
        type=float,
        help="Override dense weight for hybrid retrieval",
    )
    parser.add_argument(
        "--data-batch-size",
        type=int,
        help="Override embedding batch size for build_retrieval_candidates",
    )
    parser.add_argument(
        "--data-validate-only",
        action="store_true",
        help="Run the retrieval builder in validation-only mode",
    )
    parser.add_argument(
        "--data-allow-missed-positives",
        action="store_true",
        help="Allow builder runs to finish even if queries miss positives in top-k",
    )
    parser.add_argument(
        "--data-allow-missing-ground-truth",
        action="store_true",
        help="Allow builder runs to continue when positive filters resolve zero matches",
    )
    parser.add_argument("--device", default="auto", help="Device passed to the trainer")
    parser.add_argument("--max-train", type=int, help="Optional cap on training pairs for smoke tests")
    parser.add_argument("--max-eval", type=int, help="Optional cap on eval pairs for smoke tests")
    parser.add_argument("--trainer-batch-size", type=int, help="Override training batch size")
    parser.add_argument("--trainer-learning-rate", type=float, help="Override training learning rate")
    parser.add_argument("--trainer-max-length", type=int, help="Override training max length")
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force enable/disable mixed precision",
    )
    parser.add_argument(
        "--split",
        default=RERANK_EVAL_CONFIG.get("default_split", "train"),
        choices=["train", "test", "all"],
        help="Evaluation split (train/test/all)",
    )
    parser.add_argument("--model-path", help="Cross-encoder checkpoint to evaluate")
    parser.add_argument("--baseline-model", help="Retriever baseline path override")
    parser.add_argument("--baseline-batch-size", type=int, help="Batch size for retriever scoring")
    parser.add_argument("--cross-batch-size", type=int, help="Batch size for reranker scoring")
    parser.add_argument("--eval-output-dir", help="Custom directory for eval outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    args = parser.parse_args()
    python_exe = _python_executable()
    eval_output_dir = Path(args.eval_output_dir) if args.eval_output_dir else EVAL_OUTPUT_DIR_DEFAULT
    stats_output_path = None
    if args.stage == "full" and not args.use_legacy_triplets and not args.dry_run:
        stats_output_path = _create_stats_path()

    stages = {
        "validate": lambda: _run_data_stage(
            python_exe, args, args.dry_run, force_validate_only=True, stats_output=None
        ),
        "data": lambda: _run_data_stage(
            python_exe, args, args.dry_run, stats_output=stats_output_path if not args.use_legacy_triplets else None
        ),
        "train": lambda: _run_train_stage(python_exe, args, args.dry_run),
        "evaluate": lambda: _run_eval_stage(python_exe, args, args.dry_run, output_dir=eval_output_dir),
    }

    order = [args.stage] if args.stage != "full" else ["data", "train", "evaluate"]

    stage_durations: Dict[str, float] = {}
    retrieval_stats: Dict[str, Any] | None = None
    eval_payload: Dict[str, Any] | None = None

    try:
        for stage in order:
            success, duration = stages[stage]()
            stage_durations[stage] = duration
            if not success:
                sys.exit(1)
            if stage == "data" and stats_output_path and stats_output_path.exists():
                retrieval_stats = _load_json(stats_output_path)
            if stage == "evaluate" and not args.dry_run:
                summary_path = eval_output_dir / EVAL_SUMMARY_FILENAME
                if summary_path.exists():
                    eval_payload = _load_json(summary_path)
    finally:
        if stats_output_path and stats_output_path.exists():
            try:
                stats_output_path.unlink()
            except OSError:
                pass

    if args.stage == "full" and not args.dry_run:
        if args.use_legacy_triplets:
            logger.warning("Legacy triplet mode enabled; skipping master rerank log append.")
        elif not retrieval_stats:
            logger.warning("Retrieval stats missing; skipping master rerank log append.")
        elif not eval_payload:
            logger.warning("Evaluation summary missing; skipping master rerank log append.")
        else:
            row = _build_master_row(
                args,
                retrieval_stats=retrieval_stats,
                eval_payload=eval_payload,
                durations=stage_durations,
            )
            _append_master_log(row)

    print("\n🎉 Pipeline complete!")


if __name__ == "__main__":
    main()
