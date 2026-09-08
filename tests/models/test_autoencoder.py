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
