import json
import subprocess
from pathlib import Path

from incident_intel.runs.run_logger import log_run


def test_log_run_writes_json_with_config_metrics_and_git_commit(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_root, check=True)
    (repo_root / "file.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo_root, check=True)

    runs_dir = tmp_path / "runs"
    output_path = log_run(
        runs_dir=runs_dir,
        run_id="frame-diff-baseline",
        config={"scorer": "FrameDiffScorer"},
        metrics={"auc_roc": 0.71},
        repo_root=repo_root,
    )

    assert output_path == runs_dir / "frame-diff-baseline.json"
    record = json.loads(output_path.read_text())
    assert record["run_id"] == "frame-diff-baseline"
    assert record["config"] == {"scorer": "FrameDiffScorer"}
    assert record["metrics"] == {"auc_roc": 0.71}
    assert len(record["git_commit"]) == 40


def test_log_run_falls_back_to_unknown_outside_a_git_repo(tmp_path: Path) -> None:
    non_repo = tmp_path / "not-a-repo"
    non_repo.mkdir()

    output_path = log_run(
        runs_dir=tmp_path / "runs",
        run_id="x",
        config={},
        metrics={},
        repo_root=non_repo,
    )

    record = json.loads(output_path.read_text())
    assert record["git_commit"] == "unknown"
