from pathlib import Path
from typing import Any, Dict, Optional

from run_settings import CWD as SETTINGS_CWD, METADATA as SETTINGS_METADATA


def get_run_context(output_name: Optional[str] = None) -> Dict[str, Any]:
    cwd = Path(SETTINGS_CWD)
    return {
        "cwd": cwd,
        "metadata": dict(SETTINGS_METADATA),
        "md_to_chunk": cwd / f"{cwd.name}.md",
        "output_path": (cwd / output_name) if output_name else None,
    }
