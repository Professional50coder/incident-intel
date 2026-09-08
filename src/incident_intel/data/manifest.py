from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ClipManifest:
    clip_id: str
    split: str
    frame_count: int
    checksum: str


@dataclass
class DatasetManifest:
    dataset_name: str
    version: str
    clips: list[ClipManifest]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _sha256_of_files(paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths):
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def build_manifest(dataset_root: Path, dataset_name: str, version: str) -> DatasetManifest:
    """Scan a UCSD-Ped-style dataset root and build a checksummed manifest.

    Expects `dataset_root/Train/<clip>/*.tif` and `dataset_root/Test/<clip>/*.tif`.
    `_gt` suffixed directories under Test hold pixel-level ground-truth masks, not frame
    data, and are skipped here (ped2_dataset.py reads them separately for labels).
    """
    clips: list[ClipManifest] = []
    for split_name, split_dir in (("train", "Train"), ("test", "Test")):
        split_path = dataset_root / split_dir
        if not split_path.is_dir():
            continue
        for clip_dir in sorted(split_path.iterdir()):
            if not clip_dir.is_dir() or clip_dir.name.endswith("_gt"):
                continue
            frames = sorted(clip_dir.glob("*.tif"))
            if not frames:
                continue
            clips.append(
                ClipManifest(
                    clip_id=f"{split_name}/{clip_dir.name}",
                    split=split_name,
                    frame_count=len(frames),
                    checksum=_sha256_of_files(frames),
                )
            )
    return DatasetManifest(dataset_name=dataset_name, version=version, clips=clips)


def write_manifest(manifest: DatasetManifest, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.to_json())
