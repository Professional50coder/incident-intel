"""Turning raw per-frame anomaly scores into something a person can act on.

Pure functions, no I/O and no model: thresholding (absolute or by a false-alarm budget),
event segmentation, plain-language explanations and report rendering. Everything here is
re-computable from the stored scores, so changing the threshold never needs a re-run.

Ideas (not code) follow common practice in the VAD literature: per-clip score normalisation
with a robust (median/MAD) baseline, and event-level segmentation with gap merging.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

ABSOLUTE_THRESHOLD = 0.00047  # Ped2-calibrated reconstruction-error threshold (see README).
DEFAULT_BUDGET = 0.05  # flag at most ~5% of sampled frames in budget mode
DEFAULT_MIN_Z = 5.0  # budget mode also needs a clear bump over the clip's own baseline
MODES = ("absolute", "budget")


@dataclass
class Segment:
    start_s: float
    end_s: float
    first_sample: int
    last_sample: int
    peak_score: float
    mean_score: float
    peak_z: float
    peak_time_s: float
    explanation: str = ""
    confidence: str = "low"  # low / medium / high: how far above this clip's normal, and for how long
    rank: int = 0  # 1 = review first
    duration_s: float = 0.0


def _scale(x: np.ndarray) -> float:
    """Robust spread of a clip: 1.4826 * MAD, floored at 2% of the median so a near-static
    clip (MAD ~ 0) still has a sensible scale instead of dividing by zero."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return max(mad, 0.02 * abs(med), 1e-12)


def robust_z(scores: Sequence[float]) -> np.ndarray:
    """Per-clip robust z-score: (x - median) / robust spread."""
    x = np.asarray(scores, dtype=np.float64)
    if x.size == 0:
        return x
    return (x - float(np.median(x))) / _scale(x)


def flag_frames(
    scores: Sequence[float],
    mode: str = "absolute",
    threshold: float = ABSOLUTE_THRESHOLD,
    budget: float = DEFAULT_BUDGET,
    min_z: float = DEFAULT_MIN_Z,
) -> tuple[list[bool], float]:
    """Return (flags, effective_threshold).

    absolute: score > threshold (calibrated on Ped2 only, meaningful for that scene).
    budget:   score is in the top `budget` fraction of this clip AND at least `min_z`
              robust standard deviations above the clip median. Bounds false alarms per
              clip to roughly the budget, whatever the scene. Not validated on any
              benchmark; it is an operating heuristic.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    x = np.asarray(scores, dtype=np.float64)
    if x.size == 0:
        return [], threshold
    if mode == "absolute":
        return [bool(s > threshold) for s in x], float(threshold)
    if not 0.0 < budget <= 1.0:
        raise ValueError("budget must be in (0, 1]")
    q = float(np.quantile(x, 1.0 - budget))
    med = float(np.median(x))
    eff = max(q, med + min_z * _scale(x))
    return [bool(s >= eff and s > med) for s in x], eff


def segment_events(
    scores: Sequence[float],
    times_s: Sequence[float],
    flags: Sequence[bool],
    sample_period_s: float,
    merge_gap: int = 1,
) -> list[Segment]:
    """Group flagged samples into events, bridging gaps of up to `merge_gap` samples."""
    z = robust_z(scores)
    idx = [i for i, f in enumerate(flags) if f]
    groups: list[list[int]] = []
    for i in idx:
        if groups and i - groups[-1][-1] <= merge_gap + 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    segments = []
    for g in groups:
        lo, hi = g[0], g[-1]
        window = list(range(lo, hi + 1))
        peak = max(window, key=lambda k: scores[k])
        seg = Segment(
            start_s=float(times_s[lo]),
            end_s=float(times_s[hi]) + sample_period_s,
            first_sample=lo,
            last_sample=hi,
            peak_score=float(scores[peak]),
            mean_score=float(np.mean([scores[k] for k in window])),
            peak_z=float(z[peak]),
            peak_time_s=float(times_s[peak]),
        )
        seg.duration_s = seg.end_s - seg.start_s
        seg.confidence = confidence_for(seg.peak_z, hi - lo + 1)
        seg.explanation = explain(seg)
        segments.append(seg)
    for rank, seg in enumerate(sorted(segments, key=lambda g: (-g.peak_z, g.start_s)), 1):
        seg.rank = rank
    return segments


def confidence_for(peak_z: float, n_samples: int) -> str:
    """Heuristic label, not a calibrated probability: strong AND sustained deviations rank higher."""
    if peak_z >= 10 and n_samples >= 2:
        return "high"
    if peak_z >= 7 or (peak_z >= 5 and n_samples >= 2):
        return "medium"
    return "low"


def fmt_time(seconds: float) -> str:
    m, s = divmod(max(seconds, 0.0), 60)
    return f"{int(m)}:{s:04.1f}"


def explain(seg: Segment) -> str:
    strength = "strongly" if seg.peak_z >= 8 else "clearly" if seg.peak_z >= 4 else "moderately"
    why = {
        "high": "Strong and lasting departure, so worth checking first, though still not proof of an incident.",
        "medium": "Clear departure, but short or modest. Worth a look.",
        "low": "Weak or brief departure. Could easily be noise; check only if you have time.",
    }[seg.confidence]
    return (
        f"From {fmt_time(seg.start_s)} to {fmt_time(seg.end_s)} the scene looked {strength} "
        f"unlike the rest of this clip to the model (peak at {fmt_time(seg.peak_time_s)}). "
        "That can be a person or vehicle moving unusually, a crowd or lighting change, or "
        "camera movement; the model cannot say which. " + why
    )


def domain_warning(scores: Sequence[float], absolute_threshold: float = ABSOLUTE_THRESHOLD) -> str | None:
    """The model was trained on one walkway (UCSD Ped2). If most frames exceed the Ped2
    threshold the clip is probably a different scene, so absolute flags are unreliable."""
    x = np.asarray(scores, dtype=np.float64)
    if x.size and float(np.mean(x > absolute_threshold)) > 0.5:
        return (
            "Most frames of this clip score above the Ped2-calibrated threshold, which "
            "usually means a scene the model has never seen. Use the clip-relative mode and "
            "treat flags as the most unusual moments in this clip, not confirmed incidents."
        )
    return None


def summarize(segments: Sequence[dict], duration_s: float | None, warning: str | None, mode: str) -> dict:
    """Plain-language clip summary and a suggested review order. Honest about uncertainty."""
    notes = []
    if warning:
        notes.append("This clip looks different from the scene the model was trained on, so scores are less reliable than usual.")
    if mode == "absolute":
        notes.append("Ped2-calibrated mode is only trustworthy for the UCSD Ped2 walkway.")
    notes.append("Flags are unverified: the model has about 0.67 AUC on its benchmark and cannot name what it saw.")
    if not segments:
        return {
            "headline": "Nothing stood out in this clip at the current settings.",
            "review_order": [],
            "uncertainty": notes,
        }
    ranked = sorted(segments, key=lambda g: g["rank"])
    first = ranked[0]
    n = len(segments)
    headline = (
        f"{n} segment{'s' if n != 1 else ''} worth reviewing"
        + (f" in {fmt_time(duration_s)} of video" if duration_s else "")
        + f". Start with the one at {fmt_time(first['start_s'])} ({first['confidence']} confidence)."
    )
    return {
        "headline": headline,
        "review_order": [
            {"rank": g["rank"], "start_s": g["start_s"], "end_s": g["end_s"], "confidence": g["confidence"]}
            for g in ranked
        ],
        "uncertainty": notes,
    }


def build_report(result: dict, mode: str, threshold: float, budget: float, min_z: float = DEFAULT_MIN_Z) -> dict:
    scores = [fs["score"] for fs in result["frame_scores"]]
    times = [fs.get("time_s", float(fs["frame_index"])) for fs in result["frame_scores"]]
    period = result.get("sample_period_s") or (times[1] - times[0] if len(times) > 1 else 1.0)
    flags, eff = flag_frames(scores, mode, threshold, budget, min_z)
    segs = segment_events(scores, times, flags, period)
    seg_dicts = [asdict(g) for g in segs]
    warning = domain_warning(scores)
    return {
        "incident_id": result["incident_id"],
        "source_filename": result["source_filename"],
        "created_at": result["created_at"],
        "duration_s": result.get("duration_s"),
        "sampled_frames": len(scores),
        "truncated": result.get("truncated", False),
        "settings": {
            "mode": mode,
            "threshold": threshold,
            "budget": budget,
            "min_z": min_z,
            "effective_threshold": eff,
        },
        "flagged_frames": int(sum(flags)),
        "segments": seg_dicts,
        "warning": warning,
        "summary": summarize(seg_dicts, result.get("duration_s"), warning, mode),
        "frames": [
            {"time_s": t, "score": s, "z": float(z), "flagged": f}
            for t, s, z, f in zip(times, scores, robust_z(scores), flags)
        ],
        "model_note": (
            "Unsupervised conv autoencoder trained on UCSD Ped2 (frame-level AUC 0.674 on the "
            "official test set). Scores rank how unusual a frame looks; they are not verdicts."
        ),
    }


def report_json(report: dict) -> str:
    return json.dumps(report, indent=2)


def report_csv(report: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time_s", "score", "robust_z", "flagged"])
    for f in report["frames"]:
        w.writerow([f"{f['time_s']:.3f}", f"{f['score']:.6g}", f"{f['z']:.3f}", int(f["flagged"])])
    return buf.getvalue()


def report_markdown(report: dict) -> str:
    st = report["settings"]
    dur = fmt_time(report["duration_s"]) if report.get("duration_s") else "unknown"
    lines = [
        f"# Incident report: {report['source_filename']}",
        "",
        f"- Report id: `{report['incident_id']}`",
        f"- Analysed: {report['created_at']}",
        f"- Duration: {dur} ({report['sampled_frames']} sampled frames)",
        f"- Threshold mode: **{st['mode']}** (effective score threshold {st['effective_threshold']:.6g})",
        f"- Flagged frames: {report['flagged_frames']}; segments: {len(report['segments'])}",
        "",
    ]
    if report.get("truncated"):
        lines += ["> Only the first part of the video was analysed (length limit).", ""]
    if report.get("warning"):
        lines += [f"> **Warning:** {report['warning']}", ""]
    sm = report["summary"]
    lines += ["## Summary", "", sm["headline"], ""]
    lines += [f"- Note: {n}" for n in sm["uncertainty"]] + [""]
    lines += ["## Flagged segments", ""]
    if not report["segments"]:
        lines += ["No segments were flagged at these settings.", ""]
    else:
        lines += [
            "| Review order | Start | End | Confidence | Peak score | Peak vs clip baseline |",
            "|---|---|---|---|---|---|",
        ]
        by_rank = sorted(report["segments"], key=lambda g: g["rank"])
        for s in by_rank:
            lines.append(
                f"| {s['rank']} | {fmt_time(s['start_s'])} | {fmt_time(s['end_s'])} | {s['confidence']} | "
                f"{s['peak_score']:.6g} | {s['peak_z']:.1f} sigma |"
            )
        lines.append("")
        for s in by_rank:
            lines += [f"**Segment {s['rank']}.** {s['explanation']}", ""]
    lines += [
        "## Reviewer sign-off",
        "",
        "- [ ] Reviewed footage, confirmed incident",
        "- [ ] False alarm",
        "- Notes:",
        "",
        "## About this report",
        "",
        report["model_note"],
        "",
    ]
    return "\n".join(lines)
