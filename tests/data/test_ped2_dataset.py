from pathlib import Path

import torch
from PIL import Image

from incident_intel.config import FRAME_SIZE
from incident_intel.data.ped2_dataset import Ped2FrameDataset


def _write_frame(path: Path, fill: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=fill).save(path)


def _write_mask(path: Path, anomalous: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fill = 255 if anomalous else 0
    Image.new("L", (32, 32), color=fill).save(path)


def _build_fixture(root: Path) -> None:
    for i in range(1, 4):
        _write_frame(root / "Train" / "Train001" / f"{i:03d}.tif", fill=10 * i)

    for i in range(1, 4):
        _write_frame(root / "Test" / "Test001" / f"{i:03d}.tif", fill=10 * i)
    _write_mask(root / "Test" / "Test001_gt" / "001.bmp", anomalous=False)
    _write_mask(root / "Test" / "Test001_gt" / "002.bmp", anomalous=False)
    _write_mask(root / "Test" / "Test001_gt" / "003.bmp", anomalous=True)


def test_train_split_is_all_label_zero(tmp_path: Path) -> None:
    root = tmp_path / "UCSDped2"
    _build_fixture(root)

    dataset = Ped2FrameDataset(root, split="train")

    assert len(dataset) == 3
    for i in range(len(dataset)):
        frame, label = dataset[i]
        assert label == 0
        assert frame.shape == (1, *FRAME_SIZE)
        assert isinstance(frame, torch.Tensor)


def test_test_split_labels_come_from_gt_masks(tmp_path: Path) -> None:
    root = tmp_path / "UCSDped2"
    _build_fixture(root)

    dataset = Ped2FrameDataset(root, split="test")

    labels = [dataset[i][1] for i in range(len(dataset))]
    assert labels == [0, 0, 1]


def test_frame_values_are_normalized_to_zero_one(tmp_path: Path) -> None:
    root = tmp_path / "UCSDped2"
    _build_fixture(root)

    dataset = Ped2FrameDataset(root, split="train")
    frame, _ = dataset[0]

    assert frame.min() >= 0.0
    assert frame.max() <= 1.0
