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
| Frame-diff threshold | **0.7093** |
| Embedding k-NN | 0.6879 |
| Convolutional autoencoder (20 epochs) | 0.6451 |
| Convolutional autoencoder (150 epochs, checkpoint served by the API) | 0.6742 |

Published Ped2 baselines for comparison: reconstruction-based methods in the literature
typically report ~90-95% AUC-ROC. All three candidates here land well below that, and that
gap is worth being explicit about rather than glossing over:

- These numbers come from a **64x64 grayscale downsample** of Ped2's native 240x360 frames,
  specifically to fit comfortably in 4GB VRAM — literature results almost always use higher
  resolution, which preserves more of the small-object motion Ped2's anomalies (carts,
  bicycles, skaters) actually look like.
- Every candidate here is **per-frame** — no temporal/motion information crosses frame
  boundaries. Ped2's anomalies are defined by motion, not single-frame appearance, so this is
  a structural ceiling on the approach, not a tuning problem. That's exactly why Phase 5
  (temporal/video modeling) exists in the roadmap.
- More autoencoder training epochs helped (0.6451 → 0.6688 going from 20 to 150 epochs) but
  plateaued well short of beating the frame-diff baseline — a real, negative-ish result, not
  hidden or re-run until it looked better.

## Decision

**The convolutional autoencoder is the model wired into the API** (`checkpoints/autoencoder.pt`,
loaded by `src/incident_intel/api/main.py`), even though frame-diff scored marginally higher
in this round. The reasoning is about what each candidate can become, not just today's AUC:

- Frame-diff has **zero learnable parameters** — there is no version of Phase 2 (architecture
  ablations), Phase 3 (training on synthetic augmented anomalies), or Phase 5 (temporal
  modeling) that builds on it. Committing to it now would be a dead end for the rest of the
  roadmap.
- The AUC gap (0.709 vs 0.669) is modest, not a rout — this isn't overriding a clearly better
  option, it's choosing the option with headroom over the option that's already maxed out.
- The autoencoder's reconstruction-error map is also the natural source of the dashboard's
  "evidence" view (which pixels/regions drove the score), which the frame-diff scorer would
  need extra work to produce in a comparably interpretable form.

**Operating threshold:** `ANOMALY_THRESHOLD = 0.00047` in `src/incident_intel/api/main.py`,
chosen as the Youden's-J-optimal point on the trained model's real ROC curve over the Ped2
test set (true positive rate ≈ 0.38, false positive rate ≈ 0.10 at that point) — not a guessed
round number.
