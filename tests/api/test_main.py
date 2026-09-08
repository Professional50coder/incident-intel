from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from incident_intel.api.main import create_app


def _write_synthetic_video(path: Path, frame_count: int = 15) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (32, 32))
    rng = np.random.default_rng(seed=0)
    for _ in range(frame_count):
        frame = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _constant_score_scorer(value: float):
    def scorer(frames: torch.Tensor) -> torch.Tensor:
        return torch.full((frames.shape[0],), value)

    return scorer


def test_analyze_samples_every_fifth_frame_and_scores_each(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, frame_count=15)
    db_path = tmp_path / "test.db"

    app = create_app(scorer_factory=lambda: _constant_score_scorer(0.5), db_path=db_path)
    client = TestClient(app)

    with open(video_path, "rb") as f:
        response = client.post("/analyze", files={"file": ("clip.mp4", f, "video/mp4")})

    assert response.status_code == 200
    body = response.json()
    assert len(body["frame_scores"]) == 3  # frames 0, 5, 10 of 15 at every_nth=5
    assert body["max_score"] == 0.5
    assert body["source_filename"] == "clip.mp4"


def test_get_incident_returns_previously_analyzed_result(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, frame_count=5)
    db_path = tmp_path / "test.db"

    app = create_app(scorer_factory=lambda: _constant_score_scorer(0.1), db_path=db_path)
    client = TestClient(app)

    with open(video_path, "rb") as f:
        create_response = client.post("/analyze", files={"file": ("clip.mp4", f, "video/mp4")})
    incident_id = create_response.json()["incident_id"]

    get_response = client.get(f"/incidents/{incident_id}")

    assert get_response.status_code == 200
    assert get_response.json()["incident_id"] == incident_id


def test_get_incident_404s_for_unknown_id(tmp_path: Path) -> None:
    app = create_app(scorer_factory=lambda: _constant_score_scorer(0.1), db_path=tmp_path / "test.db")
    client = TestClient(app)

    response = client.get("/incidents/does-not-exist")

    assert response.status_code == 404


def test_list_incidents_returns_most_recent_first(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    app = create_app(scorer_factory=lambda: _constant_score_scorer(0.03), db_path=db_path)
    client = TestClient(app)

    ids = []
    for _ in range(3):
        video_path = tmp_path / f"clip-{len(ids)}.mp4"
        _write_synthetic_video(video_path, frame_count=5)
        with open(video_path, "rb") as f:
            resp = client.post("/analyze", files={"file": (video_path.name, f, "video/mp4")})
        ids.append(resp.json()["incident_id"])

    response = client.get("/incidents")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 3
    assert [item["incident_id"] for item in body] == list(reversed(ids))
    assert body[0]["is_anomalous"] is True  # score 0.03 > ANOMALY_THRESHOLD 0.02


def test_stats_are_empty_with_no_incidents(tmp_path: Path) -> None:
    app = create_app(scorer_factory=lambda: _constant_score_scorer(0.1), db_path=tmp_path / "test.db")
    client = TestClient(app)

    response = client.get("/stats")

    assert response.status_code == 200
    assert response.json() == {
        "total_incidents": 0,
        "total_frames_analyzed": 0,
        "anomalous_incident_count": 0,
        "anomaly_rate": 0.0,
        "average_max_score": 0.0,
    }


def test_stats_aggregate_across_incidents(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    app = create_app(scorer_factory=lambda: _constant_score_scorer(0.5), db_path=db_path)
    client = TestClient(app)

    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, frame_count=10)  # every_nth=5 -> 2 frames
    with open(video_path, "rb") as f:
        client.post("/analyze", files={"file": ("clip.mp4", f, "video/mp4")})

    response = client.get("/stats")

    assert response.status_code == 200
    body = response.json()
    assert body["total_incidents"] == 1
    assert body["total_frames_analyzed"] == 2
    assert body["anomalous_incident_count"] == 1
    assert body["anomaly_rate"] == 1.0
    assert body["average_max_score"] == 0.5
