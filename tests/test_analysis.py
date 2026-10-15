import numpy as np
import pytest

from incident_intel import analysis


def test_robust_z_flat_clip_is_zero():
    assert not analysis.robust_z([0.5] * 10).any()


def test_budget_mode_flags_only_clear_outliers():
    rng = np.random.default_rng(0)
    scores = list(rng.normal(1.0, 0.05, 100))
    scores[40] = 3.0
    flags, eff = analysis.flag_frames(scores, "budget", budget=0.05)
    assert [i for i, f in enumerate(flags) if f] == [40]
    assert eff > 1.2


def test_budget_mode_flags_nothing_on_flat_or_noise_only_clip():
    assert not any(analysis.flag_frames([0.5] * 50, "budget")[0])
    rng = np.random.default_rng(1)
    assert not any(analysis.flag_frames(list(rng.normal(1, 0.05, 200)), "budget")[0])


def test_absolute_mode_uses_threshold():
    flags, eff = analysis.flag_frames([0.1, 0.9], "absolute", threshold=0.5)
    assert flags == [False, True] and eff == 0.5


def test_invalid_mode_and_budget():
    with pytest.raises(ValueError):
        analysis.flag_frames([1.0], "nope")
    with pytest.raises(ValueError):
        analysis.flag_frames([1.0, 2.0], "budget", budget=0)


def test_segments_merge_small_gaps_and_split_large_ones():
    flags = [False, True, False, True, False, False, False, True]
    times = [float(i) for i in range(8)]
    segs = analysis.segment_events([1, 5, 1, 6, 1, 1, 1, 7], times, flags, 1.0)
    assert [(s.first_sample, s.last_sample) for s in segs] == [(1, 3), (7, 7)]
    assert segs[0].peak_score == 6 and segs[0].end_s == 4.0
    assert "model cannot say" in segs[0].explanation


def test_domain_warning_only_when_most_frames_exceed_threshold():
    assert analysis.domain_warning([0.001] * 10) is not None
    assert analysis.domain_warning([0.0001] * 10) is None


def _result(scores):
    return {
        "incident_id": "abc",
        "source_filename": "x.mp4",
        "created_at": "2026-01-01T00:00:00+00:00",
        "duration_s": float(len(scores)),
        "sample_period_s": 1.0,
        "frame_scores": [{"frame_index": i, "score": s, "time_s": float(i)} for i, s in enumerate(scores)],
    }


def test_reports_render_all_formats():
    scores = [1.0] * 30
    scores[10] = 20.0
    rep = analysis.build_report(_result(scores), "budget", 0.00047, 0.05)
    assert len(rep["segments"]) == 1
    assert "Segment 1" in analysis.report_markdown(rep)
    assert analysis.report_csv(rep).splitlines()[0] == "time_s,score,robust_z,flagged"
    assert '"segments"' in analysis.report_json(rep)


def test_empty_report_says_nothing_flagged():
    rep = analysis.build_report(_result([1.0] * 10), "budget", 0.00047, 0.05)
    assert "No segments were flagged" in analysis.report_markdown(rep)


def test_ranking_orders_by_strength_and_confidence_labels():
    scores = [1.0] * 40
    scores[5] = 12.0  # single weak-ish spike
    scores[25], scores[26] = 60.0, 55.0  # strong and sustained
    rep = analysis.build_report(_result(scores), "budget", 0.00047, 0.2)
    by_rank = sorted(rep["segments"], key=lambda g: g["rank"])
    assert by_rank[0]["start_s"] == 25.0 and by_rank[0]["confidence"] == "high"
    assert analysis.confidence_for(5.5, 1) == "low"
    assert analysis.confidence_for(8, 1) == "medium"


def test_summary_flags_uncertainty_and_handles_empty():
    empty = analysis.summarize([], 10.0, None, "budget")
    assert "Nothing stood out" in empty["headline"] and empty["review_order"] == []
    warned = analysis.summarize([], 10.0, "domain", "absolute")
    assert any("different from the scene" in n for n in warned["uncertainty"])
