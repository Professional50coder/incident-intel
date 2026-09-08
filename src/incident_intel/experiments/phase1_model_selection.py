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
