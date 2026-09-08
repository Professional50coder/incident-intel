from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from incident_intel.config import FRAME_SIZE


@dataclass
class FrameRecord:
    clip_id: str
    frame_path: Path
    label: int


def _load_grayscale_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L").resize(FRAME_SIZE[::-1])  # PIL wants (W, H)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


def _mask_has_anomaly(mask_path: Path) -> bool:
    mask = np.asarray(Image.open(mask_path).convert("L"))
    return bool((mask > 0).any())


def index_split(dataset_root: Path, split: str) -> list[FrameRecord]:
    """Build the list of (frame, label) records for 'train' or 'test'."""
    split_dir_name = "Train" if split == "train" else "Test"
    split_path = dataset_root / split_dir_name
    records: list[FrameRecord] = []
    for clip_dir in sorted(split_path.iterdir()):
        if not clip_dir.is_dir() or clip_dir.name.endswith("_gt"):
            continue
        gt_dir = split_path / f"{clip_dir.name}_gt"
        for frame_path in sorted(clip_dir.glob("*.tif")):
            if split == "train":
                label = 0
            else:
                mask_path = gt_dir / f"{frame_path.stem}.bmp"
                label = 1 if gt_dir.is_dir() and mask_path.exists() and _mask_has_anomaly(mask_path) else 0
            records.append(FrameRecord(clip_id=clip_dir.name, frame_path=frame_path, label=label))
    return records


class Ped2FrameDataset(Dataset):
    """Frame-level dataset over a UCSD-Ped2-style directory (see manifest.py for the layout)."""

    def __init__(self, dataset_root: Path, split: str) -> None:
        self.records = index_split(dataset_root, split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        return _load_grayscale_tensor(record.frame_path), record.label
