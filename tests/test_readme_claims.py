"""Guards the README's headline numbers against the recorded run files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _auc(name: str) -> float:
    return json.loads((ROOT / "runs" / f"phase1-{name}.json").read_text())["metrics"]["auc_roc"]


def test_readme_numbers_match_run_files():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("frame-diff", "embedding-knn", "autoencoder"):
        assert f"{_auc(name):.4f}" in readme, name


def test_served_threshold_documented_consistently():
    from incident_intel.analysis import ABSOLUTE_THRESHOLD

    assert str(ABSOLUTE_THRESHOLD) in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "0.00047" in (ROOT / "docs/model-selection/phase-1.md").read_text(encoding="utf-8")


def test_checkpoint_loads_and_runs():
    import torch

    from incident_intel.models.autoencoder import load_checkpoint, reconstruction_error

    model = load_checkpoint(ROOT / "checkpoints" / "autoencoder.pt")
    assert reconstruction_error(model, torch.rand(2, 1, 64, 64)).shape == (2,)
