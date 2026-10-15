from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from incident_intel.api.main import create_app


def _video(path: Path, frames: int = 40, spike_at: Optional[int] = 20) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    for i in range(frames):
        img = np.full((32, 32, 3), 100, np.uint8)
        if spike_at is not None and spike_at <= i < spike_at + 5:
            img[:] = 250
        w.write(img)
    w.release()


def _brightness_scorer(frames: torch.Tensor) -> torch.Tensor:
    return frames.flatten(1).mean(1) * 0.001  # bright frames score high


def _client(tmp_path: Path, **kw) -> TestClient:
    return TestClient(create_app(scorer_factory=lambda: _brightness_scorer, db_path=tmp_path / "t.db", **kw))


def test_index_and_health_and_config(tmp_path):
    c = _client(tmp_path)
    assert "Try a sample" in c.get("/").text
    assert "Analyse a clip" in c.get("/app").text
    assert c.get("/static/og.png").status_code == 200
    assert c.get("/robots.txt").status_code == 200
    assert c.get("/health").json()["status"] == "ok"
    cfg = c.get("/config").json()
    assert ".mp4" in cfg["allowed_extensions"] and cfg["max_upload_mb"] > 0


def test_frames_carry_real_time_and_report_formats(tmp_path):
    v = tmp_path / "a.mp4"
    _video(v)
    c = _client(tmp_path)
    r = c.post("/analyze?mode=budget", files={"file": ("a.mp4", v.read_bytes(), "video/mp4")}).json()
    assert r["frame_scores"][1]["source_frame"] == 5 and abs(r["frame_scores"][1]["time_s"] - 0.5) < 0.05
    assert abs(r["duration_s"] - 4.0) < 0.2
    rep = c.get(f"/incidents/{r['incident_id']}/report?mode=budget").json()
    assert len(rep["segments"]) == 1 and 1.5 < rep["segments"][0]["start_s"] < 2.6
    assert c.get(f"/incidents/{r['incident_id']}/report?format=csv").text.startswith("time_s,")
    md = c.get(f"/incidents/{r['incident_id']}/report?format=md&download=true")
    assert "attachment" in md.headers["content-disposition"] and "Segment 1" in md.text
    assert c.get(f"/incidents/{r['incident_id']}/report?format=pdf").status_code == 422
    assert c.get("/incidents/nope/report").status_code == 404


def test_upload_limits_and_type_checks(tmp_path):
    c = _client(tmp_path, max_upload_mb=0.001)
    assert c.post("/analyze", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 415
    assert c.post("/analyze", files={"file": ("a.mp4", b"x" * 5000, "video/mp4")}).status_code == 413
    assert c.post("/analyze?mode=bogus", files={"file": ("a.mp4", b"x", "video/mp4")}).status_code == 422
    assert c.post("/analyze?budget=5", files={"file": ("a.mp4", b"x", "video/mp4")}).status_code == 422


def test_history_is_scoped_per_client_and_deletable(tmp_path):
    v = tmp_path / "a.mp4"
    _video(v)
    c = _client(tmp_path)
    mine, other = {"X-Client-Id": "client-aaaaaaaa"}, {"X-Client-Id": "client-bbbbbbbb"}
    r = c.post("/analyze", files={"file": ("a.mp4", v.read_bytes(), "video/mp4")}, headers=mine).json()
    assert len(c.get("/incidents", headers=mine).json()) == 1
    assert c.get("/incidents", headers=other).json() == []
    assert c.get("/stats", headers=other).json()["total_incidents"] == 0
    assert c.delete(f"/incidents/{r['incident_id']}", headers=other).status_code == 404
    assert c.delete(f"/incidents/{r['incident_id']}", headers=mine).status_code == 204
    assert c.get(f"/incidents/{r['incident_id']}").status_code == 404


def test_filename_is_sanitised(tmp_path):
    v = tmp_path / "a.mp4"
    _video(v, frames=10)
    c = _client(tmp_path)
    r = c.post("/analyze", files={"file": ("../../<script>x.mp4", v.read_bytes(), "video/mp4")}).json()
    assert "/" not in r["source_filename"] and "<" not in r["source_filename"]


def test_samples_list_analyze_and_unknown(tmp_path):
    sd = tmp_path / "samples"
    sd.mkdir()
    _video(sd / "demo.mp4")
    (sd / "demo.txt").write_text("A demo clip.")
    c = _client(tmp_path, samples_dir=sd)
    assert c.get("/samples").json()[0]["description"] == "A demo clip."
    assert c.get("/samples/demo/video").status_code == 200
    r = c.post("/samples/demo/analyze")
    assert r.status_code == 200 and r.json()["source_filename"] == "demo.mp4"
    assert c.post("/samples/..%2Fetc/analyze").status_code == 404
    assert c.post("/samples/missing/analyze").status_code == 404
    assert _client(tmp_path / "x", samples_dir=tmp_path / "none").get("/samples").json() == []


def test_rate_limit_returns_429_with_retry_after(tmp_path):
    c = _client(tmp_path, rate_limit_per_min=2)
    for _ in range(2):
        assert c.post("/analyze", files={"file": ("a.txt", b"x", "text/plain")}).status_code == 415
    r = c.post("/analyze", files={"file": ("a.txt", b"x", "text/plain")})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert "Too many" in r.json()["detail"]
    # other clients (different forwarded address) are unaffected; cheap endpoints are not limited
    assert c.post("/analyze", files={"file": ("a.txt", b"x", "text/plain")}, headers={"X-Forwarded-For": "9.9.9.9"}).status_code == 415
    assert c.get("/health").status_code == 200


def test_report_has_ranked_segments_and_plain_summary(tmp_path):
    v = tmp_path / "a.mp4"
    _video(v)
    c = _client(tmp_path)
    r = c.post("/analyze?mode=budget", files={"file": ("a.mp4", v.read_bytes(), "video/mp4")}).json()
    rep = c.get(f"/incidents/{r['incident_id']}/report").json()
    seg = rep["segments"][0]
    assert seg["rank"] == 1 and seg["confidence"] in {"low", "medium", "high"}
    assert "Start with the one at" in rep["summary"]["headline"]
    assert any("unverified" in n for n in rep["summary"]["uncertainty"])
    assert "## Summary" in c.get(f"/incidents/{r['incident_id']}/report?format=md").text
