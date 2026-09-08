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
