from __future__ import annotations

import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List

import cv2
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from incident_intel.config import FRAME_SIZE
from incident_intel.models.autoencoder import ConvAutoencoder, load_checkpoint, reconstruction_error

DB_PATH = Path("data/incident_intel.db")
CHECKPOINT_PATH = Path("checkpoints/autoencoder.pt")
ANOMALY_THRESHOLD = 0.02  # set from Task 8's Phase 1 evaluation - see docs/model-selection/phase-1.md
EVERY_NTH_FRAME = 5

FrameScorer = Callable[[torch.Tensor], torch.Tensor]  # (batch, 1, H, W) -> (batch,) scores


class FrameScoreOut(BaseModel):
    frame_index: int
    score: float
    is_anomalous: bool


class IncidentResult(BaseModel):
    incident_id: str
    created_at: str
    source_filename: str
    frame_scores: List[FrameScoreOut]
    max_score: float


class IncidentSummary(BaseModel):
    incident_id: str
    created_at: str
    source_filename: str
    max_score: float
    is_anomalous: bool
    frame_count: int


class DashboardStats(BaseModel):
    total_incidents: int
    total_frames_analyzed: int
    anomalous_incident_count: int
    anomaly_rate: float
    average_max_score: float


def _default_scorer() -> FrameScorer:
    """Lazily loads the checkpoint on first use, not at import time - the module (and its
    `app` instance below) must stay importable even before a checkpoint exists on disk, e.g.
    during test collection or before Task 8's training run has produced one."""
    state: dict[str, ConvAutoencoder] = {}

    def score(frames: torch.Tensor) -> torch.Tensor:
        if "model" not in state:
            state["model"] = load_checkpoint(CHECKPOINT_PATH)
        return reconstruction_error(state["model"], frames)

    return score


def _sample_frames_from_video(path: Path, every_nth: int = EVERY_NTH_FRAME) -> List[torch.Tensor]:
    cap = cv2.VideoCapture(str(path))
    frames: List[torch.Tensor] = []
    index = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if index % every_nth == 0:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, FRAME_SIZE[::-1])
                tensor = torch.from_numpy(resized.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0)
                frames.append(tensor)
            index += 1
    finally:
        cap.release()
    return frames


def _get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            source_filename TEXT NOT NULL,
            result_json TEXT NOT NULL
        )
        """
    )
    return conn


def create_app(
    scorer_factory: Callable[[], FrameScorer] = _default_scorer,
    db_path: Path = DB_PATH,
) -> FastAPI:
    """Builds the API with an injectable scorer/db path, so tests don't need a trained
    checkpoint on disk."""
    app = FastAPI(title="Incident Intelligence API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    scorer = scorer_factory()

    @app.post("/analyze", response_model=IncidentResult)
    async def analyze(file: UploadFile = File(...)) -> IncidentResult:
        raw = await file.read()
        suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        try:
            frames = _sample_frames_from_video(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        if not frames:
            raise HTTPException(status_code=400, detail="no frames could be read from the uploaded video")

        frame_scores: List[FrameScoreOut] = []
        for i, frame in enumerate(frames):
            score = float(scorer(frame)[0])
            frame_scores.append(
                FrameScoreOut(frame_index=i, score=score, is_anomalous=score > ANOMALY_THRESHOLD)
            )

        result = IncidentResult(
            incident_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            source_filename=file.filename or "unknown",
            frame_scores=frame_scores,
            max_score=max(fs.score for fs in frame_scores),
        )

        conn = _get_connection(db_path)
        with conn:
            conn.execute(
                "INSERT INTO incidents (id, created_at, source_filename, result_json) VALUES (?, ?, ?, ?)",
                (result.incident_id, result.created_at, result.source_filename, result.model_dump_json()),
            )
        conn.close()
        return result

    @app.get("/incidents/{incident_id}", response_model=IncidentResult)
    async def get_incident(incident_id: str) -> IncidentResult:
        conn = _get_connection(db_path)
        row = conn.execute(
            "SELECT result_json FROM incidents WHERE id = ?", (incident_id,)
        ).fetchone()
        conn.close()
        if row is None:
            raise HTTPException(status_code=404, detail="incident not found")
        return IncidentResult.model_validate_json(row[0])

    @app.get("/incidents", response_model=List[IncidentSummary])
    async def list_incidents(limit: int = 50) -> List[IncidentSummary]:
        conn = _get_connection(db_path)
        rows = conn.execute(
            "SELECT result_json FROM incidents ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        summaries = []
        for (result_json,) in rows:
            result = IncidentResult.model_validate_json(result_json)
            summaries.append(
                IncidentSummary(
                    incident_id=result.incident_id,
                    created_at=result.created_at,
                    source_filename=result.source_filename,
                    max_score=result.max_score,
                    is_anomalous=any(fs.is_anomalous for fs in result.frame_scores),
                    frame_count=len(result.frame_scores),
                )
            )
        return summaries

    @app.get("/stats", response_model=DashboardStats)
    async def get_stats() -> DashboardStats:
        conn = _get_connection(db_path)
        rows = conn.execute("SELECT result_json FROM incidents").fetchall()
        conn.close()
        if not rows:
            return DashboardStats(
                total_incidents=0,
                total_frames_analyzed=0,
                anomalous_incident_count=0,
                anomaly_rate=0.0,
                average_max_score=0.0,
            )
        results = [IncidentResult.model_validate_json(row[0]) for row in rows]
        anomalous_count = sum(1 for r in results if any(fs.is_anomalous for fs in r.frame_scores))
        return DashboardStats(
            total_incidents=len(results),
            total_frames_analyzed=sum(len(r.frame_scores) for r in results),
            anomalous_incident_count=anomalous_count,
            anomaly_rate=anomalous_count / len(results),
            average_max_score=sum(r.max_score for r in results) / len(results),
        )

    return app


app = create_app()
