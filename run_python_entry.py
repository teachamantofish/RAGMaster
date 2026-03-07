#!/usr/bin/env python3
"""Run repository Python scripts with a consistent import context.

This launcher is the single entry point for subprocess-based script execution.
It guarantees that repo root is on ``sys.path`` and executes target scripts
as ``__main__`` so behavior matches direct invocation.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _resolve_script(script_arg: str) -> Path:
    candidate = Path(script_arg)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    resolved = candidate.resolve()

    repo_norm = os.path.normcase(str(REPO_ROOT))
    script_norm = os.path.normcase(str(resolved))
    if not script_norm.startswith(repo_norm + os.sep) and script_norm != repo_norm:
        raise ValueError(f"Script path must be inside repository root: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Script file not found: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a repo script with repo-root imports.")
    parser.add_argument("script", help="Script path relative to repo root, or absolute path inside repo")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to target script")
    args = parser.parse_args()

    target = _resolve_script(args.script)
    launch_paths = [str(target.parent), str(REPO_ROOT)]
    for entry in reversed(launch_paths):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    sys.argv = [str(target), *args.script_args]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
