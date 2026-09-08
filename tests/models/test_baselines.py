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
