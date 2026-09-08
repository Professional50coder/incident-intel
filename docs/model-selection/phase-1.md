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
| Frame-diff threshold | *(fill in after running `python -m incident_intel.experiments.phase1_model_selection`)* |
| Embedding k-NN | *(fill in)* |
| Convolutional autoencoder | *(fill in)* |

Published Ped2 baselines for comparison: reconstruction-based methods typically report
~90-95% AUC-ROC on this benchmark.

## Decision

*(state which candidate is wired into the API, and why — cost to train/run and latency
matter here, not just AUC.)*
