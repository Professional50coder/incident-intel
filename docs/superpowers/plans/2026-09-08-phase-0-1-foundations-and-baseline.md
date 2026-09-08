# Phase 0+1: Foundations & Baseline Anomaly Detector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data engineering pipeline, a rigorously-compared set of baseline anomaly detectors, and a working FastAPI backend that scores uploaded video for anomalies — the first fully working, testable increment of Incident Intelligence.

**Architecture:** A `src/incident_intel` Python package with clearly separated `data` (manifest + dataset loading), `models` (baseline scorers + the trained autoencoder), `eval` (metrics), `runs` (experiment logging), and `api` (FastAPI) layers. Model selection is a real, run-and-recorded comparison (frame-diff threshold vs. pretrained-embedding k-NN vs. trained autoencoder) before any model is wired into the API. The frontend (Next.js + Three.js dashboard) is deliberately **out of scope for this plan** — it's a separate subsystem with its own design skills and depends on this API existing first; it gets its own plan once this one ships.

**Tech Stack:** Python 3.10+, PyTorch + torchvision (CUDA 12.4 build, compatible with the local RTX 3050's driver), FastAPI, SQLite (stdlib `sqlite3`), OpenCV (`opencv-python-headless`), Pillow, NumPy, scikit-learn (AUC-ROC), pytest + httpx (API tests).

**Spec:** `docs/spec/2026-09-08-project-spec.md`

## Global Constraints

- All models and batch sizes must fit in **4GB VRAM** (RTX 3050 Laptop) — frames are downsized to 64x64 grayscale specifically to keep this comfortable.
- No cloud services, no telemetry-shipping tools (no W&B/MLflow) — experiment tracking is local JSON under `runs/`.
- No proprietary data — UCSD Ped2 only, a public benchmark.
- Every dataset used by an experiment must be traceable to a manifest entry (`src/incident_intel/data/manifest.py`) — no ad hoc, unversioned data loading.
- venv lives at `D:\venvs\incident-intel` — never the OneDrive-synced Desktop, never the PATH/hermes venv.

---

### Task 1: Environment & repo scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `README.md`
- Create: `.paul/PROJECT.md`
- Create: `.paul/STATE.md`
- Create: `src/incident_intel/__init__.py`
- Create: `src/incident_intel/config.py`

**Interfaces:**
- Produces: `incident_intel.config.FRAME_SIZE: tuple[int, int]` — `(64, 64)`, the `(H, W)` every frame is resized to. Every later task that touches a frame tensor imports this constant instead of hardcoding it.

This task has no unit-visible behavior of its own — it's scaffolding the rest of the plan needs — so its "test" is a real smoke check instead of a pytest run.

- [ ] **Step 1: Create the venv**

```bash
python -m venv /d/venvs/incident-intel
source /d/venvs/incident-intel/Scripts/activate
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "incident-intel"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Write `requirements.txt`**

torch/torchvision are installed separately (Step 4) because they need a CUDA-specific index URL that doesn't fit a plain `requirements.txt` line.

```text
fastapi
uvicorn[standard]
python-multipart
pydantic
opencv-python-headless
pillow
numpy
scikit-learn
pytest
httpx
```

- [ ] **Step 4: Install dependencies**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

- [ ] **Step 5: Write `src/incident_intel/__init__.py`** (empty file, makes the package importable)

- [ ] **Step 6: Write `src/incident_intel/config.py`**

```python
"""Project-wide constants. Import from here instead of hardcoding values that must stay
consistent across data loading, training, and serving (e.g. frame size)."""

FRAME_SIZE: tuple[int, int] = (64, 64)  # (H, W) - downsized from Ped2's native 240x360 to
                                          # keep training comfortably inside 4GB VRAM.
```

- [ ] **Step 7: Write `.gitignore`**

```text
__pycache__/
*.pyc
.pytest_cache/
data/raw/
data/*.db
checkpoints/
*.egg-info/
```

- [ ] **Step 8: Write `README.md`**

```markdown
# Incident Intelligence

Video anomaly detection portfolio project. See `docs/spec/2026-09-08-project-spec.md` for the
full design and phase map, and `.paul/PROJECT.md` / `.paul/STATE.md` for current status.

## Setup

\`\`\`bash
python -m venv /d/venvs/incident-intel
source /d/venvs/incident-intel/Scripts/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
\`\`\`

## Tests

\`\`\`bash
pytest
\`\`\`
```

- [ ] **Step 9: Write `.paul/PROJECT.md`**

```markdown
# Incident Intelligence — PAUL Project

**Spec:** `docs/spec/2026-09-08-project-spec.md`

## Milestones (phases)

0. Foundations — repo, data pipeline, venv, PAUL setup
1. Baseline product — model-selection comparison, trained autoencoder, FastAPI backend
2. Model training & experimentation
3. Generative augmentation
4. Explanation layer (VLM)
5. Temporal/video modeling
6. Training engineering
7. Distributed training study
8. Productionization & UI polish

Phase 1's Next.js + Three.js dashboard is planned separately from the ML/API work above
(see STATE.md) since it's an independent subsystem with its own design-skill workflow.
```

- [ ] **Step 10: Write `.paul/STATE.md`**

```markdown
# State

**Current milestone:** Phase 0 + Phase 1 (backend/ML) — see
`docs/superpowers/plans/2026-09-08-phase-0-1-foundations-and-baseline.md`

**Status:** In progress.

**Next up after this plan ships:** Phase 1 frontend (Next.js + Three.js dashboard) as its own
plan, then Phase 2 (model training & experimentation).
```

- [ ] **Step 11: Verify the environment**

```bash
python -c "import torch, cv2, fastapi; print('cuda available:', torch.cuda.is_available())"
python -c "import incident_intel; print(incident_intel.config.FRAME_SIZE)"
```

Expected: no import errors, `cuda available: True`, prints `(64, 64)`.

- [ ] **Step 12: Commit**

```bash
git add pyproject.toml requirements.txt .gitignore README.md .paul src
git commit -m "chore: scaffold incident-intel project"
```

---

### Task 2: Dataset manifest (data engineering versioning)

**Files:**
- Create: `src/incident_intel/data/__init__.py`
- Create: `src/incident_intel/data/manifest.py`
- Test: `tests/data/test_manifest.py`

**Interfaces:**
- Consumes: nothing from earlier tasks besides the package scaffold.
- Produces: `ClipManifest` (dataclass: `clip_id: str`, `split: str`, `frame_count: int`, `checksum: str`), `DatasetManifest` (dataclass: `dataset_name: str`, `version: str`, `clips: list[ClipManifest]`, method `to_json() -> str`), `build_manifest(dataset_root: Path, dataset_name: str, version: str) -> DatasetManifest`, `write_manifest(manifest: DatasetManifest, output_path: Path) -> None`. Task 3 and Task 8 both call `build_manifest`.

Expects a UCSD-Ped-style layout: `dataset_root/Train/<clip>/*.tif` (normal-only) and
`dataset_root/Test/<clip>/*.tif` with matching `dataset_root/Test/<clip>_gt/*.bmp` pixel masks
(confirmed structure per Anomalib's UCSD Ped docs and the dataset's own distribution).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_manifest.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'incident_intel.data.manifest'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/incident_intel/data/manifest.py
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
    data, and are skipped here (Task 3 reads them separately for labels).
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
```

Also create `src/incident_intel/data/__init__.py` (empty).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_manifest.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/incident_intel/data/__init__.py src/incident_intel/data/manifest.py tests/data/test_manifest.py
git commit -m "feat: add dataset manifest for versioned data engineering"
```

---

### Task 3: Ped2 frame dataset loader

**Files:**
- Create: `src/incident_intel/data/ped2_dataset.py`
- Test: `tests/data/test_ped2_dataset.py`

**Interfaces:**
- Consumes: `incident_intel.config.FRAME_SIZE` (Task 1).
- Produces: `FrameRecord` (dataclass: `clip_id: str`, `frame_path: Path`, `label: int`), `index_split(dataset_root: Path, split: str) -> list[FrameRecord]`, `Ped2FrameDataset(dataset_root: Path, split: str)` — a `torch.utils.data.Dataset` whose `__getitem__` returns `(frame: torch.Tensor of shape (1, H, W), label: int)`. Task 4, 5, 8 all construct `Ped2FrameDataset` directly.

Frame-level labels for the test split come from the matching `<clip>_gt/<frame>.bmp` pixel
mask: label is 1 if the mask has any non-zero pixel, else 0. Train-split frames are always
label 0 (the dataset guarantees every training clip is anomaly-free).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_ped2_dataset.py
from pathlib import Path

import numpy as np
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_ped2_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'incident_intel.data.ped2_dataset'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/incident_intel/data/ped2_dataset.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_ped2_dataset.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/incident_intel/data/ped2_dataset.py tests/data/test_ped2_dataset.py
git commit -m "feat: load Ped2 frames with gt-mask-derived labels"
```

---

### Task 4: Baseline anomaly scorers (model-selection candidates 1 & 2)

**Files:**
- Create: `src/incident_intel/models/__init__.py`
- Create: `src/incident_intel/models/baselines.py`
- Test: `tests/models/test_baselines.py`

**Interfaces:**
- Consumes: nothing beyond `torch`.
- Produces: `FrameDiffScorer` (methods `fit(normal_frames: torch.Tensor) -> None`, `score(frames: torch.Tensor) -> torch.Tensor`), `EmbeddingKNNScorer(encoder: torch.nn.Module, k: int = 5)` (same `fit`/`score` shape). `frames`/`normal_frames` are `(N, 1, H, W)`; `score` returns `(N,)`. Task 8's model-selection script instantiates both.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_baselines.py
import torch
import torch.nn as nn

from incident_intel.models.baselines import EmbeddingKNNScorer, FrameDiffScorer


def test_frame_diff_scorer_scores_distance_from_normal_average() -> None:
    normal = torch.zeros(4, 1, 8, 8)
    scorer = FrameDiffScorer()
    scorer.fit(normal)

    same_as_normal = torch.zeros(1, 1, 8, 8)
    very_different = torch.ones(1, 1, 8, 8)

    assert scorer.score(same_as_normal).item() == 0.0
    assert scorer.score(very_different).item() == 1.0


def test_frame_diff_scorer_requires_fit_first() -> None:
    scorer = FrameDiffScorer()
    try:
        scorer.score(torch.zeros(1, 1, 8, 8))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_embedding_knn_scorer_ranks_similar_frame_below_dissimilar_one() -> None:
    # A tiny, deterministic "encoder" - just global-average-pools each channel - so the test
    # doesn't need to download pretrained weights.
    encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
    scorer = EmbeddingKNNScorer(encoder=encoder, k=1)

    normal_bank = torch.zeros(3, 1, 8, 8)
    scorer.fit(normal_bank)

    close_frame = torch.zeros(1, 1, 8, 8) + 0.01
    far_frame = torch.ones(1, 1, 8, 8)

    close_score = scorer.score(close_frame).item()
    far_score = scorer.score(far_frame).item()
    assert close_score < far_score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'incident_intel.models.baselines'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/incident_intel/models/baselines.py
from __future__ import annotations

import torch
import torch.nn as nn


class FrameDiffScorer:
    """Non-learned baseline: score = mean absolute pixel difference from the average normal frame."""

    def __init__(self) -> None:
        self._reference: torch.Tensor | None = None

    def fit(self, normal_frames: torch.Tensor) -> None:
        self._reference = normal_frames.mean(dim=0, keepdim=True)

    def score(self, frames: torch.Tensor) -> torch.Tensor:
        if self._reference is None:
            raise RuntimeError("call fit() before score()")
        diff = (frames - self._reference).abs()
        return diff.flatten(start_dim=1).mean(dim=1)


class EmbeddingKNNScorer:
    """Pretrained-embedding baseline: score = mean distance to the k nearest normal-frame
    embeddings. `encoder` is expected frozen (eval mode, no grad needed)."""

    def __init__(self, encoder: nn.Module, k: int = 5) -> None:
        self._encoder = encoder.eval()
        self._k = k
        self._bank: torch.Tensor | None = None

    @torch.no_grad()
    def _embed(self, frames: torch.Tensor) -> torch.Tensor:
        rgb = frames.repeat(1, 3, 1, 1) if frames.shape[1] == 1 else frames
        return self._encoder(rgb).flatten(start_dim=1)

    def fit(self, normal_frames: torch.Tensor) -> None:
        self._bank = self._embed(normal_frames)

    def score(self, frames: torch.Tensor) -> torch.Tensor:
        if self._bank is None:
            raise RuntimeError("call fit() before score()")
        embeddings = self._embed(frames)
        dists = torch.cdist(embeddings, self._bank)
        nearest_k = dists.topk(min(self._k, self._bank.shape[0]), largest=False).values
        return nearest_k.mean(dim=1)
```

Also create `src/incident_intel/models/__init__.py` (empty).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_baselines.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/incident_intel/models/__init__.py src/incident_intel/models/baselines.py tests/models/test_baselines.py
git commit -m "feat: add frame-diff and embedding-kNN baseline anomaly scorers"
```

---

### Task 5: Convolutional autoencoder (model-selection candidate 3)

**Files:**
- Create: `src/incident_intel/models/autoencoder.py`
- Test: `tests/models/test_autoencoder.py`

**Interfaces:**
- Consumes: `incident_intel.config.FRAME_SIZE` (Task 1).
- Produces: `ConvAutoencoder` (`nn.Module`, `forward(x: (N,1,H,W)) -> (N,1,H,W)`), `reconstruction_error(model: ConvAutoencoder, frames: torch.Tensor) -> torch.Tensor` (shape `(N,)`), `save_checkpoint(model, path: Path) -> None`, `load_checkpoint(path: Path, device: str = "cpu") -> ConvAutoencoder`. Task 6's training script and Task 9's API both use `reconstruction_error` and `load_checkpoint`.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_autoencoder.py
from pathlib import Path

import torch

from incident_intel.config import FRAME_SIZE
from incident_intel.models.autoencoder import (
    ConvAutoencoder,
    load_checkpoint,
    reconstruction_error,
    save_checkpoint,
)


def test_forward_preserves_shape() -> None:
    model = ConvAutoencoder()
    x = torch.rand(2, 1, *FRAME_SIZE)

    out = model(x)

    assert out.shape == x.shape


def test_reconstruction_error_is_nonnegative_and_per_frame() -> None:
    model = ConvAutoencoder()
    x = torch.rand(3, 1, *FRAME_SIZE)

    scores = reconstruction_error(model, x)

    assert scores.shape == (3,)
    assert torch.all(scores >= 0)


def test_checkpoint_roundtrip_preserves_output(tmp_path: Path) -> None:
    model = ConvAutoencoder()
    model.eval()
    x = torch.rand(1, 1, *FRAME_SIZE)
    original_output = model(x)

    checkpoint_path = tmp_path / "autoencoder.pt"
    save_checkpoint(model, checkpoint_path)
    loaded = load_checkpoint(checkpoint_path)
    loaded_output = loaded(x)

    assert torch.allclose(original_output, loaded_output)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_autoencoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'incident_intel.models.autoencoder'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/incident_intel/models/autoencoder.py
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Small convolutional autoencoder for 64x64 grayscale frames, trained on normal frames
    only. Reconstruction error on unseen frames is the anomaly score - fits comfortably in
    4GB VRAM and trains in minutes."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def reconstruction_error(model: ConvAutoencoder, frames: torch.Tensor) -> torch.Tensor:
    """Per-frame anomaly score: mean-squared reconstruction error, shape (batch,)."""
    model.eval()
    with torch.no_grad():
        recon = model(frames)
        return ((recon - frames) ** 2).flatten(start_dim=1).mean(dim=1)


def save_checkpoint(model: ConvAutoencoder, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(path: Path, device: str = "cpu") -> ConvAutoencoder:
    model = ConvAutoencoder()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_autoencoder.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/incident_intel/models/autoencoder.py tests/models/test_autoencoder.py
git commit -m "feat: add convolutional autoencoder anomaly model"
```

---

### Task 6: AUC-ROC evaluation metric

**Files:**
- Create: `src/incident_intel/eval/__init__.py`
- Create: `src/incident_intel/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: nothing beyond `scikit-learn`.
- Produces: `compute_auc_roc(labels: Sequence[int], scores: Sequence[float]) -> float`. Task 8's model-selection script and Task 8's write-up both call this for every candidate scorer.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_metrics.py
import pytest

from incident_intel.eval.metrics import compute_auc_roc


def test_perfect_separation_gives_auc_one() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    assert compute_auc_roc(labels, scores) == pytest.approx(1.0)


def test_inverted_scores_give_auc_zero() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.9, 0.8, 0.2, 0.1]

    assert compute_auc_roc(labels, scores) == pytest.approx(0.0)


def test_single_class_labels_raise_value_error() -> None:
    with pytest.raises(ValueError):
        compute_auc_roc([1, 1, 1], [0.1, 0.5, 0.9])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'incident_intel.eval.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/incident_intel/eval/metrics.py
from __future__ import annotations

from typing import Sequence

from sklearn.metrics import roc_auc_score


def compute_auc_roc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """AUC-ROC for anomaly scores against binary frame-level ground truth (1 = anomalous)."""
    if len(set(labels)) < 2:
        raise ValueError("AUC-ROC is undefined when all labels are the same class")
    return float(roc_auc_score(labels, scores))
```

Also create `src/incident_intel/eval/__init__.py` (empty).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/incident_intel/eval tests/eval/test_metrics.py
git commit -m "feat: add AUC-ROC evaluation metric"
```

---

### Task 7: Experiment run logger

**Files:**
- Create: `src/incident_intel/runs/__init__.py`
- Create: `src/incident_intel/runs/run_logger.py`
- Test: `tests/runs/test_run_logger.py`

**Interfaces:**
- Consumes: nothing beyond stdlib + git.
- Produces: `log_run(runs_dir: Path, run_id: str, config: dict, metrics: dict, repo_root: Path) -> Path`. Task 8's model-selection script calls this once per candidate scorer.

- [ ] **Step 1: Write the failing test**

```python
# tests/runs/test_run_logger.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/runs/test_run_logger.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'incident_intel.runs.run_logger'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/incident_intel/runs/run_logger.py
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
```

Also create `src/incident_intel/runs/__init__.py` (empty).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/runs/test_run_logger.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/incident_intel/runs tests/runs/test_run_logger.py
git commit -m "feat: add local JSON experiment run logger"
```

---

### Task 8: Model-selection experiment + write-up (real dataset)

This is the task that actually needs UCSD Ped2 downloaded, and produces the honest,
documented model-selection comparison the spec calls for. Unlike Tasks 1-7, its "test" is
running the script against real data and recording what it actually prints — there is no
way to pre-write real AUC numbers before the code runs.

**Files:**
- Create: `data/manifests/.gitkeep`
- Create: `src/incident_intel/experiments/__init__.py`
- Create: `src/incident_intel/experiments/phase1_model_selection.py`
- Test: `tests/experiments/test_phase1_model_selection.py`
- Create: `docs/model-selection/phase-1.md`
- Create: `checkpoints/.gitkeep`

**Interfaces:**
- Consumes: `build_manifest`/`write_manifest` (Task 2), `Ped2FrameDataset` (Task 3),
  `FrameDiffScorer`/`EmbeddingKNNScorer` (Task 4), `ConvAutoencoder`/`reconstruction_error`/
  `save_checkpoint` (Task 5), `compute_auc_roc` (Task 6), `log_run` (Task 7).
- Produces: `run_model_selection(dataset_root: Path, checkpoint_out: Path) -> dict[str, float]`
  (maps scorer name to AUC-ROC), plus the trained autoencoder checkpoint on disk that Task 9's
  API loads.

**Manual prerequisite (do this before Step 1):** Download UCSD Ped2. The dataset's original
host (`svcl.ucsd.edu`) is not always reachable; the Kaggle mirror
`karthiknm1/ucsd-anomaly-detection-dataset` is a reliable alternative. Extract it so you end
up with `data/raw/UCSDped2/Train/...` and `data/raw/UCSDped2/Test/...` matching the layout
Task 2/3 expect (frames as `.tif`, test ground truth as `<clip>_gt/*.bmp`). This directory is
gitignored (Task 1's `.gitignore`) — only the manifest and results get committed, not the
1GB+ of raw frames.

**Methodology note (goes into the write-up, not hidden):** UCSD Ped2 only has 12 test clips
total, so there's no large separate validation split to tune on without touching the same
data literature reports on. Candidates are compared on the full official test set, exactly as
published baselines do, and this is stated explicitly in the write-up as a known limitation of
using a small, over-studied benchmark for model selection — not glossed over.

- [ ] **Step 1: Write the failing test** (uses synthetic frames, not the real dataset, so it
  runs fast and offline — the real-dataset run happens in Step 5)

```python
# tests/experiments/test_phase1_model_selection.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/experiments/test_phase1_model_selection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'incident_intel.experiments.phase1_model_selection'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/incident_intel/experiments/phase1_model_selection.py
from __future__ import annotations

from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

from incident_intel.data.manifest import build_manifest, write_manifest
from incident_intel.data.ped2_dataset import Ped2FrameDataset
from incident_intel.eval.metrics import compute_auc_roc
from incident_intel.models.autoencoder import ConvAutoencoder, reconstruction_error, save_checkpoint
from incident_intel.models.baselines import EmbeddingKNNScorer, FrameDiffScorer
from incident_intel.runs.run_logger import log_run


def _stack(dataset: Ped2FrameDataset) -> tuple[torch.Tensor, torch.Tensor]:
    frames = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    return frames, labels


def run_model_selection(
    dataset_root: Path,
    checkpoint_out: Path,
    epochs: int = 20,
    repo_root: Path | None = None,
) -> dict[str, float]:
    repo_root = repo_root or Path(__file__).resolve().parents[3]
    manifest = build_manifest(dataset_root, dataset_name="UCSDped2", version="v1")
    write_manifest(manifest, repo_root / "data" / "manifests" / "ucsdped2-v1.json")

    train_ds = Ped2FrameDataset(dataset_root, split="train")
    test_ds = Ped2FrameDataset(dataset_root, split="test")
    train_frames, _ = _stack(train_ds)
    test_frames, test_labels = _stack(test_ds)
    labels_list = test_labels.tolist()

    results: dict[str, float] = {}

    # Candidate 1: non-learned frame-diff baseline.
    frame_diff = FrameDiffScorer()
    frame_diff.fit(train_frames)
    frame_diff_scores = frame_diff.score(test_frames)
    results["frame_diff"] = compute_auc_roc(labels_list, frame_diff_scores.tolist())
    log_run(
        repo_root / "runs",
        run_id="phase1-frame-diff",
        config={"scorer": "FrameDiffScorer"},
        metrics={"auc_roc": results["frame_diff"]},
        repo_root=repo_root,
    )

    # Candidate 2: pretrained-embedding k-NN baseline.
    encoder = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
    encoder.classifier = torch.nn.Identity()
    knn = EmbeddingKNNScorer(encoder=encoder, k=5)
    knn.fit(train_frames)
    knn_scores = knn.score(test_frames)
    results["embedding_knn"] = compute_auc_roc(labels_list, knn_scores.tolist())
    log_run(
        repo_root / "runs",
        run_id="phase1-embedding-knn",
        config={"scorer": "EmbeddingKNNScorer", "encoder": "mobilenet_v2", "k": 5},
        metrics={"auc_roc": results["embedding_knn"]},
        repo_root=repo_root,
    )

    # Candidate 3: trained convolutional autoencoder.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for frames, _labels in loader:
            frames = frames.to(device)
            optimizer.zero_grad()
            recon = model(frames)
            loss = loss_fn(recon, frames)
            loss.backward()
            optimizer.step()
    save_checkpoint(model, checkpoint_out)

    ae_scores = reconstruction_error(model.to("cpu"), test_frames)
    results["autoencoder"] = compute_auc_roc(labels_list, ae_scores.tolist())
    log_run(
        repo_root / "runs",
        run_id="phase1-autoencoder",
        config={"scorer": "ConvAutoencoder", "epochs": epochs},
        metrics={"auc_roc": results["autoencoder"]},
        repo_root=repo_root,
    )

    print("Phase 1 model-selection results:")
    for name, auc in results.items():
        print(f"  {name}: AUC-ROC = {auc:.4f}")

    return results


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/UCSDped2")
    run_model_selection(root, Path("checkpoints/autoencoder.pt"))
```

Also create `src/incident_intel/experiments/__init__.py` (empty), `data/manifests/.gitkeep`,
and `checkpoints/.gitkeep`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/experiments/test_phase1_model_selection.py -v`
Expected: PASS. Note: `torchvision.models.mobilenet_v2(weights=...DEFAULT)` downloads
pretrained ImageNet weights on first run (one-time, cached under `~/.cache/torch`) — this
needs network access once, not per-test-run.

- [ ] **Step 5: Run against the real dataset and write up the results**

```bash
python -m incident_intel.experiments.phase1_model_selection data/raw/UCSDped2
```

Copy the printed AUC-ROC table verbatim into `docs/model-selection/phase-1.md` using this
template (fill in the `## Results` section with the actual numbers printed — do not invent
numbers, and if a candidate underperforms the frame-diff baseline, say so plainly):

```markdown
# Phase 1 Model Selection

## Candidates compared

1. **Frame-diff threshold** — non-learned; score = distance from the mean normal frame.
2. **Embedding k-NN** — frozen pretrained MobileNetV2 embeddings, score = mean distance to
   the 5 nearest normal-frame embeddings.
3. **Convolutional autoencoder** — trained from scratch on normal frames only; score =
   reconstruction error.

## Methodology note

UCSD Ped2 has only 12 test clips total, too few to carve out a separate validation split
without reusing data the published baselines also evaluate on. All three candidates are
scored on the full official test set, matching how the literature reports Ped2 numbers, and
that reuse is a known limitation of model-selecting on a small benchmark — not hidden.

## Results

| Candidate | AUC-ROC |
|---|---|
| Frame-diff threshold | *(fill in)* |
| Embedding k-NN | *(fill in)* |
| Convolutional autoencoder | *(fill in)* |

Published Ped2 baselines for comparison: reconstruction-based methods typically report
~90-95% AUC-ROC on this benchmark.

## Decision

*(state which candidate is wired into the API in Task 9, and why — cost to train/run and
latency matter here, not just AUC. If the autoencoder wins, say so; if a baseline wins, that's
a legitimate outcome worth reporting honestly too.)*
```

- [ ] **Step 6: Commit**

```bash
git add data/manifests src/incident_intel/experiments tests/experiments docs/model-selection checkpoints/.gitkeep
git commit -m "feat: run Phase 1 model-selection comparison and pick the baseline model"
```

---

### Task 9: FastAPI backend serving the chosen model

**Files:**
- Create: `src/incident_intel/api/__init__.py`
- Create: `src/incident_intel/api/main.py`
- Test: `tests/api/test_main.py`

**Interfaces:**
- Consumes: `ConvAutoencoder`/`reconstruction_error`/`load_checkpoint` (Task 5), the checkpoint
  written by Task 8 at `checkpoints/autoencoder.pt`.
- Produces: `create_app(scorer_factory=..., db_path=...) -> FastAPI` (injectable for tests),
  module-level `app` (the production instance). Endpoints: `POST /analyze` (multipart video
  upload → `IncidentResult`), `GET /incidents/{incident_id}` (→ `IncidentResult` or 404). The
  Phase 1 frontend plan (separate, not in this plan) consumes these two endpoints.

**Threshold note:** `ANOMALY_THRESHOLD` is set from Task 8's evaluation run, not guessed —
after Task 8 finishes, read the per-frame score distribution it computed and pick a threshold
that matches the operating point you want (e.g. the score at the ROC curve's best
Youden's-J point). Until Task 8's real numbers exist, this task's tests use an injected
threshold, so the exact production value can be filled in without blocking this task.

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_main.py
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from incident_intel.api.main import create_app


def _write_synthetic_video(path: Path, frame_count: int = 15) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (32, 32))
    rng = np.random.default_rng(seed=0)
    for _ in range(frame_count):
        frame = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _constant_score_scorer(value: float):
    def scorer(frames: torch.Tensor) -> torch.Tensor:
        return torch.full((frames.shape[0],), value)

    return scorer


def test_analyze_samples_every_fifth_frame_and_scores_each(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, frame_count=15)
    db_path = tmp_path / "test.db"

    app = create_app(scorer_factory=lambda: _constant_score_scorer(0.5), db_path=db_path)
    client = TestClient(app)

    with open(video_path, "rb") as f:
        response = client.post("/analyze", files={"file": ("clip.mp4", f, "video/mp4")})

    assert response.status_code == 200
    body = response.json()
    assert len(body["frame_scores"]) == 3  # frames 0, 5, 10 of 15 at every_nth=5
    assert body["max_score"] == 0.5
    assert body["source_filename"] == "clip.mp4"


def test_get_incident_returns_previously_analyzed_result(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, frame_count=5)
    db_path = tmp_path / "test.db"

    app = create_app(scorer_factory=lambda: _constant_score_scorer(0.1), db_path=db_path)
    client = TestClient(app)

    with open(video_path, "rb") as f:
        create_response = client.post("/analyze", files={"file": ("clip.mp4", f, "video/mp4")})
    incident_id = create_response.json()["incident_id"]

    get_response = client.get(f"/incidents/{incident_id}")

    assert get_response.status_code == 200
    assert get_response.json()["incident_id"] == incident_id


def test_get_incident_404s_for_unknown_id(tmp_path: Path) -> None:
    app = create_app(scorer_factory=lambda: _constant_score_scorer(0.1), db_path=tmp_path / "test.db")
    client = TestClient(app)

    response = client.get("/incidents/does-not-exist")

    assert response.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'incident_intel.api.main'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/incident_intel/api/main.py
from __future__ import annotations

import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List

import cv2
import torch
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from incident_intel.config import FRAME_SIZE
from incident_intel.models.autoencoder import load_checkpoint, reconstruction_error

DB_PATH = Path("data/incident_intel.db")
CHECKPOINT_PATH = Path("checkpoints/autoencoder.pt")
ANOMALY_THRESHOLD = 0.02  # set from Task 8's Phase 1 evaluation - see docs/model-selection/phase-1.md
EVERY_NTH_FRAME = 5

FrameScorer = Callable[[torch.Tensor], torch.Tensor]  # (batch, 1, H, W) -> (batch,) scores


class FrameScoreOut(BaseModel):
    frame_index: int
    score: float
    is_anomalous: bool


class IncidentResult(BaseModel):
    incident_id: str
    created_at: str
    source_filename: str
    frame_scores: List[FrameScoreOut]
    max_score: float


def _default_scorer() -> FrameScorer:
    model = load_checkpoint(CHECKPOINT_PATH)

    def score(frames: torch.Tensor) -> torch.Tensor:
        return reconstruction_error(model, frames)

    return score


def _sample_frames_from_video(path: Path, every_nth: int = EVERY_NTH_FRAME) -> List[torch.Tensor]:
    cap = cv2.VideoCapture(str(path))
    frames: List[torch.Tensor] = []
    index = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if index % every_nth == 0:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, FRAME_SIZE[::-1])
                tensor = torch.from_numpy(resized.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0)
                frames.append(tensor)
            index += 1
    finally:
        cap.release()
    return frames


def _get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            source_filename TEXT NOT NULL,
            result_json TEXT NOT NULL
        )
        """
    )
    return conn


def create_app(
    scorer_factory: Callable[[], FrameScorer] = _default_scorer,
    db_path: Path = DB_PATH,
) -> FastAPI:
    """Builds the API with an injectable scorer/db path, so tests don't need a trained
    checkpoint on disk."""
    app = FastAPI(title="Incident Intelligence API")
    scorer = scorer_factory()

    @app.post("/analyze", response_model=IncidentResult)
    async def analyze(file: UploadFile = File(...)) -> IncidentResult:
        raw = await file.read()
        suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        try:
            frames = _sample_frames_from_video(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        if not frames:
            raise HTTPException(status_code=400, detail="no frames could be read from the uploaded video")

        frame_scores: List[FrameScoreOut] = []
        for i, frame in enumerate(frames):
            score = float(scorer(frame)[0])
            frame_scores.append(
                FrameScoreOut(frame_index=i, score=score, is_anomalous=score > ANOMALY_THRESHOLD)
            )

        result = IncidentResult(
            incident_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            source_filename=file.filename or "unknown",
            frame_scores=frame_scores,
            max_score=max(fs.score for fs in frame_scores),
        )

        conn = _get_connection(db_path)
        with conn:
            conn.execute(
                "INSERT INTO incidents (id, created_at, source_filename, result_json) VALUES (?, ?, ?, ?)",
                (result.incident_id, result.created_at, result.source_filename, result.model_dump_json()),
            )
        conn.close()
        return result

    @app.get("/incidents/{incident_id}", response_model=IncidentResult)
    async def get_incident(incident_id: str) -> IncidentResult:
        conn = _get_connection(db_path)
        row = conn.execute(
            "SELECT result_json FROM incidents WHERE id = ?", (incident_id,)
        ).fetchone()
        conn.close()
        if row is None:
            raise HTTPException(status_code=404, detail="incident not found")
        return IncidentResult.model_validate_json(row[0])

    return app


app = create_app()
```

Also create `src/incident_intel/api/__init__.py` (empty).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_main.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: PASS (every test from Tasks 2-9)

- [ ] **Step 6: Commit**

```bash
git add src/incident_intel/api tests/api
git commit -m "feat: serve video anomaly analysis over FastAPI"
```

---

## After this plan

Update `.paul/STATE.md` to mark Phase 0+1 backend/ML complete, note the actual AUC-ROC numbers
and the chosen model from Task 8, and record that the Phase 1 frontend (Next.js + Three.js
dashboard, using this session's `frontend-design`/`ui-ux-pro-max` skills) is the next plan to
write — it consumes `POST /analyze` and `GET /incidents/{id}` exactly as defined in Task 9.
