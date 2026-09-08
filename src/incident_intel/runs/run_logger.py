from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    git_commit: str
    config: dict[str, Any]
    metrics: dict[str, float]


def _current_git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def log_run(
    runs_dir: Path,
    run_id: str,
    config: dict[str, Any],
    metrics: dict[str, float],
    repo_root: Path,
) -> Path:
    """Write one experiment run as a JSON file under `runs_dir`. Returns the file path."""
    record = RunRecord(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=_current_git_commit(repo_root),
        config=config,
        metrics=metrics,
    )
    runs_dir.mkdir(parents=True, exist_ok=True)
    output_path = runs_dir / f"{run_id}.json"
    output_path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True))
    return output_path
