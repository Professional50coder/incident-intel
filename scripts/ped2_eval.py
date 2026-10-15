"""Re-verify the benchmark on UCSD Ped2 with a stated protocol (run on a GPU box or CPU).

Reports, for each scorer, three frame-level AUCs so numbers can be compared with papers that
use different conventions:
  micro      all test frames pooled into one ROC (what the repo has always reported)
  macro      mean of per-video AUCs (videos with both classes only)
  micro_norm per-clip min-max normalised scores, then pooled (a common protocol choice)

Scorers: the committed checkpoint (mean-MSE and top-k pixel error), the frame-diff baseline,
and optionally N freshly trained autoencoders (seeds) to show run-to-run variance.

Usage: python scripts/ped2_eval.py DATASET_ROOT [--checkpoint checkpoints/autoencoder.pt]
                                    [--retrain-epochs 150 --seeds 3] [--out result.json]
DATASET_ROOT is the folder containing Train/ and Test/ (e.g. .../UCSDped2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from incident_intel.data.ped2_dataset import Ped2FrameDataset
from incident_intel.models.autoencoder import ConvAutoencoder, load_checkpoint, reconstruction_error
from incident_intel.models.baselines import FrameDiffScorer


def stack(ds):
    frames = torch.stack([ds[i][0] for i in range(len(ds))])
    labels = np.array([r.label for r in ds.records])
    clips = np.array([r.clip_id for r in ds.records])
    return frames, labels, clips


def topk_error(model, frames, frac=0.01):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), 256):
            x = frames[i:i + 256]
            err = ((model(x) - x) ** 2).flatten(1)
            k = max(1, int(err.shape[1] * frac))
            out.append(err.topk(k, dim=1).values.mean(1))
    return torch.cat(out).numpy()


def metrics(scores, labels, clips, ref_threshold=None):
    scores = np.asarray(scores, dtype=np.float64)
    micro = roc_auc_score(labels, scores)
    per = []
    norm = scores.copy()
    for c in np.unique(clips):
        m = clips == c
        s = scores[m]
        norm[m] = (s - s.min()) / (s.max() - s.min() + 1e-12)
        if len(set(labels[m])) == 2:
            per.append(roc_auc_score(labels[m], s))
    fpr, tpr, thr = roc_curve(labels, scores)
    j = int(np.argmax(tpr - fpr))
    out = {
        "micro_auc": float(micro),
        "macro_auc": float(np.mean(per)),
        "macro_clips": len(per),
        "micro_norm_auc": float(roc_auc_score(labels, norm)),
        "youden_threshold": float(thr[j]),
        "youden_tpr": float(tpr[j]),
        "youden_fpr": float(fpr[j]),
    }
    if ref_threshold is not None:
        pred = scores > ref_threshold
        out["at_ref_threshold"] = {
            "threshold": ref_threshold,
            "tpr": float(pred[labels == 1].mean()),
            "fpr": float(pred[labels == 0].mean()),
        }
    return out


def train_ae(train_ds, epochs, seed, device):
    torch.manual_seed(seed)
    model = ConvAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for x, _ in loader:
            x = x.to(device)
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(x), x).backward()
            opt.step()
    return model.cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/autoencoder.pt"))
    ap.add_argument("--retrain-epochs", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("ped2_eval.json"))
    args = ap.parse_args()

    train_ds = Ped2FrameDataset(args.dataset_root, "train")
    test_ds = Ped2FrameDataset(args.dataset_root, "test")
    train_x, _, _ = stack(train_ds)
    test_x, labels, clips = stack(test_ds)
    result = {
        "dataset": {
            "train_frames": len(train_ds), "test_frames": len(test_ds),
            "test_clips": int(len(np.unique(clips))), "anomalous_test_frames": int(labels.sum()),
        },
    }
    ref = 0.00047

    ck = load_checkpoint(args.checkpoint)
    result["checkpoint_mean_mse"] = metrics(reconstruction_error(ck, test_x).numpy(), labels, clips, ref)
    result["checkpoint_topk_pixel_error"] = metrics(topk_error(ck, test_x), labels, clips)

    fd = FrameDiffScorer()
    fd.fit(train_x)
    result["frame_diff"] = metrics(fd.score(test_x).numpy(), labels, clips)

    if args.retrain_epochs:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        runs = []
        for seed in range(args.seeds):
            m = train_ae(train_ds, args.retrain_epochs, seed, device)
            runs.append({
                "seed": seed,
                "mean_mse": metrics(reconstruction_error(m, test_x).numpy(), labels, clips),
                "topk_pixel_error": metrics(topk_error(m, test_x), labels, clips),
            })
        result["retrained"] = {"epochs": args.retrain_epochs, "device": device, "runs": runs}

    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
