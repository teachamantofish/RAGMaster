"""Minimal pipeline runner that executes the core scripts in sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

TRAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAIN_DIR.parent
RUNNER_SCRIPT = PROJECT_ROOT / "run_python_entry.py"
SCRIPT_SEQUENCE: Sequence[Sequence[str]] = (
    ("1create_training_data.py",),
    ("2tokenize_triplets.py",),
    ("4embedmodel_finetuner.py", "--action", "train"),
    ("5diagnostic_embeddings.py",),
    ("6evaluate_model.py",),
)


def _run_step(step_cmd: Sequence[str], dry_run: bool) -> None:
    repo_script = str((TRAIN_DIR / step_cmd[0]).relative_to(PROJECT_ROOT))
    cmd = [sys.executable, str(RUNNER_SCRIPT), repo_script, *step_cmd[1:]]
    display = " ".join(cmd)
    if dry_run:
        print(f"[DRY RUN] {display}")
        return
    print(f"▶ Running: {display}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the embedding pipeline scripts sequentially.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the commands that would run without executing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for step in SCRIPT_SEQUENCE:
        _run_step(step, dry_run=args.dry_run)
    if args.dry_run:
        print("Dry run complete; no scripts executed.")
    else:
        print("✅ Pipeline complete.")


if __name__ == "__main__":
    main()
