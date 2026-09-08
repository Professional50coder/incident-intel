# Incident Intelligence

Video anomaly detection, built end to end: a versioned data pipeline, a real model-selection
comparison across three candidate approaches, a trained PyTorch model, an evaluation against
published benchmarks, a served FastAPI backend, and a Next.js + Three.js dashboard — not a
notebook, a running system.

**The problem it solves:** given a video clip from a fixed camera (a walkway, a production
line, a storefront), flag which frames look anomalous, show *why* (a live 3D score timeline,
not just a number), and track that across a growing history of analyzed footage.

## Architecture

```
 video upload
      │
      ▼
 frame sampling (OpenCV, every 5th frame)
      │
      ▼
 anomaly scorer (trained conv autoencoder → reconstruction error)
      │
      ▼
 SQLite (incident history)          FastAPI ── /analyze  /incidents  /stats
      │                                              │
      └──────────────────────────────────────────────┘
                                                       │
                                     Next.js + Three.js dashboard
                                     (KPI tiles, live incident feed,
                                      3D score-timeline visualization)
```

## Approach

**Data engineering.** Every dataset used is versioned through a manifest
(`src/incident_intel/data/manifest.py`) that checksums each clip, so any experiment can be
traced back to exactly which frames produced it — not an ad hoc `glob` call.

**Model selection, done as a real comparison.** Before committing to a model, three candidates
were built and evaluated on the same held-out test set, on equal footing:

| Candidate | AUC-ROC |
|---|---|
| Frame-diff threshold (non-learned) | **0.709** |
| Pretrained-embedding k-NN (frozen MobileNetV2) | 0.688 |
| Convolutional autoencoder (trained, 150 epochs) | 0.674 |

The full write-up — including *why* the autoencoder was still the one wired into the API
despite scoring lower — is in [`docs/model-selection/phase-1.md`](docs/model-selection/phase-1.md).
Short version: it's the only candidate with learnable capacity, which every later phase
(training ablations, synthetic-data augmentation, temporal modeling) builds on; a frame-diff
threshold has nowhere left to go.

**Honest results, not inflated ones.** These AUC numbers sit below the ~90-95% the anomaly
detection literature reports on this benchmark. That gap is explained, not hidden: frames are
downsampled to 64x64 grayscale to fit comfortably in 4GB of VRAM, and every candidate here
scores single frames independently — no motion information crosses frame boundaries, which is
a real structural limit for a benchmark whose anomalies (cyclists, carts) are defined by
motion. Closing that gap with a temporal model is a planned later phase, not a hidden problem.

**Evaluation threshold** is set from the trained model's actual ROC curve (Youden's J
statistic), not a guessed round number.

## Tech stack

- **Backend:** Python, PyTorch + torchvision, FastAPI, SQLite, OpenCV, scikit-learn
- **Frontend:** Next.js (App Router), TypeScript, Tailwind CSS, Three.js
- **Testing:** pytest (24 tests, TDD throughout — see `tests/`)
- **Dataset:** [UCSD Ped2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html), a public
  video anomaly detection benchmark

## Repo structure

```
src/incident_intel/
  data/          dataset manifest + Ped2 frame loader
  models/        baseline scorers + the trained autoencoder
  eval/          AUC-ROC evaluation
  runs/          local experiment logging (no external telemetry service)
  experiments/   the Phase 1 model-selection script
  api/           FastAPI backend
tests/           mirrors src/, one test file per module
dashboard/       Next.js + Three.js frontend
docs/
  spec/                  full 8-phase project design
  model-selection/       per-phase model comparisons and decisions
  superpowers/plans/     detailed implementation plan for this phase
checkpoints/     trained model weights (small enough to check in)
```

## Running it locally

**Backend**

```bash
python -m venv venv && source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
pytest                                                 # 24 tests
uvicorn incident_intel.api.main:app --reload           # http://localhost:8000
```

To retrain and reproduce the model-selection numbers above, download
[UCSD Ped2](http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz) into
`data/raw/`, then:

```bash
python -m incident_intel.experiments.phase1_model_selection data/raw/UCSD_Anomaly_Dataset.v1p2/UCSDped2
```

**Dashboard**

```bash
cd dashboard
npm install
cp .env.local.example .env.local   # point NEXT_PUBLIC_API_URL at the backend
npm run dev                        # http://localhost:3000
```

## Roadmap

This is Phase 1 of an 8-phase plan (full detail in
[`docs/spec/2026-09-08-project-spec.md`](docs/spec/2026-09-08-project-spec.md)):
data engineering & baseline (done) → model training & experimentation → generative
augmentation for rare anomalies → a VLM explanation layer → temporal/video modeling →
formal training-engineering/experiment tracking → a distributed-training study →
productionization. Each phase ships as a working increment of the same running product, not a
disconnected notebook.
