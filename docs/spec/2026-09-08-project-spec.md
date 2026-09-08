# Incident Intelligence — Project Spec

## What this is

A portfolio-grade video incident intelligence system: upload footage, get back risk-scored,
evidence-backed incident detections with human-readable explanations. Built as one coherent
product (data → training → evaluation → serving → UI), not a set of disconnected notebooks.
Purpose: demonstrate the full ML lifecycle for GenPeach-style research-engineering roles.

**Constraints that shape every phase:**
- Hardware: one local machine, **NVIDIA RTX 3050 Laptop, 4GB VRAM**. No rented/cloud compute.
  Every model and batch size must fit in 4GB; where a technique (diffusion, VLM, distributed
  training) is normally done at a scale that doesn't fit, the phase is *not skipped* — it's
  scoped down honestly and the write-up says so explicitly. No fabricated numbers.
- Pace: a few hours a week. Phases are sized to finish in 2-3 weekends each, not months.
- Nothing about the *build process itself* touches Claude's cloud tooling (no Artifacts, no
  uploads/exports) — that's a constraint on how this conversation works, not on the product's
  own architecture.
- No proprietary data. Public, well-documented benchmark datasets only.

## Domain & dataset

Video anomaly detection in surveillance/monitoring footage. Dataset: **UCSD Ped2** (fallback:
CUHK Avenue if Ped2's mirror is unreliable) — short low-res clips of a pedestrian walkway,
pre-split into normal-only training clips and mixed test clips with frame-level ground-truth
anomaly labels. Small (hundreds of MB), decades-old published baselines (~90-95% AUC-ROC) to
compare against, genuinely video (not stills), and the standard technique — reconstruction-based
anomaly detection — is cheap to train and produces a natural "evidence" visualization for free
(the reconstruction-error map *is* the explanation).

## Phase map

Each phase is a PAUL milestone (see `.paul/`). A phase is not "done" until: the model-selection
comparison for that phase is written down, the data-engineering step is reproducible from a
script (not manual), and the result is wired into the running product, not left in a notebook.

| # | Phase | Core question it answers |
|---|---|---|
| 0 | Foundations | Repo, data pipeline, venv, CI-lite, PAUL setup |
| 1 | Baseline product | Does an end-to-end video → risk score → evidence pipeline work at all? |
| 2 | Model training & experimentation | Can we beat the baseline with a trained model, and prove it with ablations? |
| 3 | Generative augmentation | Can synthetic anomalies (small diffusion/GAN) improve detection on rare failure types? |
| 4 | Explanation layer (VLM) | Can the system explain *why* in plain language, not just a score? |
| 5 | Temporal/video modeling | Does modeling motion across frames beat per-frame reconstruction? |
| 6 | Training engineering | Is every experiment reproducible (config, seed, commit, metrics, checkpoint)? |
| 7 | Distributed training study | What would change with >1 GPU — studied honestly on 1 GPU |
| 8 | Productionization & UI polish | Does this look and feel like a real product, not a demo script? |

### Phase 0 — Foundations
- `D:\venvs\incident-intel` venv (never OneDrive, never the PATH/hermes venv).
- Data engineering pipeline v1: download script, frame extraction (OpenCV), a `manifest.json`
  per dataset version (checksum, frame count, split) so any later run can be traced back to
  exactly which data produced it. This is the seed of "data engineering process," not a one-off
  script — every later phase's dataset changes go through the same manifest step.
- `runs/` convention: one JSON per experiment (git commit, config, seed, metrics). No
  MLflow/W&B — those ship telemetry to their cloud.
- PAUL initialized (`.paul/PROJECT.md`, `STATE.md`), this spec linked from it.

### Phase 1 — Baseline product
- **Model selection, done as a real comparison, not a foregone conclusion**: before committing
  to the convolutional autoencoder, write a short comparison in
  `docs/model-selection/phase-1.md` scoring 2-3 candidates (e.g. plain per-pixel frame-diff
  threshold, a pretrained-CNN-embedding + k-NN distance, and the trained autoencoder) on a
  held-out slice, and state why the winner was chosen — cost to train, latency, and AUC all
  matter, not just AUC.
- Small convolutional autoencoder, trained only on normal frames (fits 4GB, minutes to train).
- Anomaly score = per-frame reconstruction error; evaluated as AUC-ROC against ground truth,
  reported next to the published baseline for the same dataset.
- Backend: FastAPI (`POST /analyze`, `GET /incidents/{id}`), SQLite for v1.
- Frontend: **Next.js + TypeScript + Tailwind**, using this session's `frontend-design` /
  `ui-ux-pro-max` skills for real visual design (not default-looking scaffolding) — score
  timeline, flagged-frame gallery, click-through evidence view. **Three.js** used where it adds
  real value, not decoration: a 3D score-over-time ribbon the user can scrub/rotate, and/or a
  particle-field view of the embedding space showing normal frames clustered vs. flagged frames
  drifting away from the cluster — makes the "why is this anomalous" story visually intuitive
  instead of a bare line chart.

### Phase 2 — Model training & experimentation
- Compare architectures head-to-head (e.g. plain conv-AE vs. a ViT-based encoder-decoder vs.
  conv-AE + augmentation), logged as a real ablation table in `docs/model-selection/phase-2.md`
  — same format as the pasted plan's experiment table (model, dataset version, F1/AUC, GPU
  memory, throughput).
- Reuses the Phase 0 manifest/versioning so each experiment cites an exact dataset version.

### Phase 3 — Generative augmentation
- A small DDPM-style diffusion model at low resolution (fits 4GB at, e.g., 32x32-64x64 crops) —
  or a lighter GAN if diffusion training time doesn't fit the weekly time budget; that choice
  itself is written up as a model-selection call, not silently picked.
- Purpose is explicit and evaluated: generate synthetic rare-anomaly frames, retrain the Phase 2
  winner with augmented data, and report whether AUC actually improves — a negative result here
  is still a legitimate, reportable finding.

### Phase 4 — Explanation layer (VLM)
- A small open, locally-runnable vision-language component (e.g. a lightweight
  captioning/CLIP-based model, quantized if needed to fit 4GB) turns a flagged frame + its
  reconstruction error into a plain-language explanation, surfaced in the dashboard next to the
  score.

### Phase 5 — Temporal/video modeling
- Replace per-frame reconstruction with a short-window temporal model (e.g. ConvLSTM or a small
  temporal transformer over N consecutive frames) to catch motion-based anomalies a single frame
  misses; compared against the Phase 1/2 per-frame baseline on the same eval set.

### Phase 6 — Training engineering
- Formalizes what Phases 1-5 already did ad hoc: every run's config/seed/commit/metrics/
  checkpoint captured consistently, surfaced as a simple local experiments view in the
  dashboard (reads the `runs/` JSON files — no new external tool).

### Phase 7 — Distributed training study
- Honest framing: with one GPU, we can't produce real multi-GPU scaling numbers, and this spec
  will not fabricate them. Instead: implement the training loop to be DDP-correct, verify it
  runs correctly with `torch.distributed` using 2 CPU processes (`gloo` backend) as a
  correctness check, and use gradient accumulation as the practical local substitute for a
  larger effective batch size. The write-up documents what would need to change for real
  multi-GPU (NCCL backend, per-GPU batch sizing, gradient sync cost) as analysis, clearly
  labeled as unverified-on-this-hardware.

### Phase 8 — Productionization & UI polish
- Docker Compose packaging (pattern the user already runs in `ppc-crm`), Postgres replacing
  SQLite, a full pass with the design skills on the dashboard (motion/interaction polish via
  `hyperframes-animation` or `frontend-design` as appropriate), and a recorded demo walkthrough
  for the portfolio. Whether this gets deployed anywhere public (a VPS, a domain) is a decision
  for that phase, not assumed now.

## What "done" looks like for the whole project

A GitHub repo where each phase is a real commit history showing: a stated hypothesis, a
model-selection comparison, a data-engineering step that's a script not a one-off, a result
(including negative results), and a product that actually got better because of it — walkable
end-to-end in an interview.
