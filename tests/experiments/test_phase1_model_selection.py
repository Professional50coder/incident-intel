from pathlib import Path

from PIL import Image

from incident_intel.experiments.phase1_model_selection import run_model_selection


def _write_frame(path: Path, fill: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=fill).save(path)


def _write_mask(path: Path, anomalous: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=255 if anomalous else 0).save(path)


def _build_synthetic_dataset(root: Path) -> None:
    for clip in ("Train001", "Train002"):
        for i in range(1, 6):
            _write_frame(root / "Train" / clip / f"{i:03d}.tif", fill=10)

    for i in range(1, 6):
        fill = 200 if i > 3 else 10  # last two frames are visually different -> "anomalous"
        _write_frame(root / "Test" / "Test001" / f"{i:03d}.tif", fill=fill)
        _write_mask(root / "Test" / "Test001_gt" / f"{i:03d}.bmp", anomalous=i > 3)


def test_run_model_selection_returns_auc_for_every_candidate(tmp_path: Path) -> None:
    dataset_root = tmp_path / "UCSDped2"
    _build_synthetic_dataset(dataset_root)
    checkpoint_out = tmp_path / "checkpoints" / "autoencoder.pt"

    results = run_model_selection(dataset_root, checkpoint_out, epochs=1)

    assert set(results.keys()) == {"frame_diff", "embedding_knn", "autoencoder"}
    for auc in results.values():
        assert 0.0 <= auc <= 1.0
    assert checkpoint_out.exists()
