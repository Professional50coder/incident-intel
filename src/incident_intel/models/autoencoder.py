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
