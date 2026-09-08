from pathlib import Path

from PIL import Image

from incident_intel.data.manifest import build_manifest, write_manifest


def _write_tif(path: Path, fill: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), color=fill).save(path)


def test_build_manifest_finds_train_and_test_clips_and_skips_gt_dirs(tmp_path: Path) -> None:
    root = tmp_path / "UCSDped2"
    _write_tif(root / "Train" / "Train001" / "001.tif", fill=10)
    _write_tif(root / "Train" / "Train001" / "002.tif", fill=20)
    _write_tif(root / "Test" / "Test001" / "001.tif", fill=30)
    _write_tif(root / "Test" / "Test001_gt" / "001.bmp", fill=0)  # must NOT be treated as a clip

    manifest = build_manifest(root, dataset_name="UCSDped2", version="v1")

    clip_ids = {c.clip_id for c in manifest.clips}
    assert clip_ids == {"train/Train001", "test/Test001"}
    train_clip = next(c for c in manifest.clips if c.clip_id == "train/Train001")
    assert train_clip.frame_count == 2
    assert train_clip.split == "train"


def test_build_manifest_checksum_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "UCSDped2"
    _write_tif(root / "Train" / "Train001" / "001.tif", fill=42)

    first = build_manifest(root, dataset_name="UCSDped2", version="v1")
    second = build_manifest(root, dataset_name="UCSDped2", version="v1")

    assert first.clips[0].checksum == second.clips[0].checksum
    assert len(first.clips[0].checksum) == 64  # sha256 hex digest


def test_write_manifest_creates_readable_json(tmp_path: Path) -> None:
    root = tmp_path / "UCSDped2"
    _write_tif(root / "Train" / "Train001" / "001.tif", fill=1)
    manifest = build_manifest(root, dataset_name="UCSDped2", version="v1")

    output_path = tmp_path / "manifests" / "ucsdped2-v1.json"
    write_manifest(manifest, output_path)

    assert output_path.exists()
    assert "UCSDped2" in output_path.read_text()
