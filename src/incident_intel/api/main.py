from __future__ import annotations

import os
import re
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import torch
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel

from incident_intel import analysis
from incident_intel.config import FRAME_SIZE
from incident_intel.models.autoencoder import ConvAutoencoder, load_checkpoint, reconstruction_error

DB_PATH = Path(os.environ.get("INCIDENT_DB", "data/incident_intel.db"))
CHECKPOINT_PATH = Path("checkpoints/autoencoder.pt")
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", "samples"))
STATIC_DIR = Path(__file__).parent / "static"
# Youden's-J point on the trained autoencoder's Ped2 test-set ROC (figures recorded in
# docs/model-selection/phase-1.md). Only meaningful for the Ped2 scene.
ANOMALY_THRESHOLD = analysis.ABSOLUTE_THRESHOLD
EVERY_NTH_FRAME = 5

MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "30"))
MAX_RAW_FRAMES = int(os.environ.get("MAX_RAW_FRAMES", "6000"))  # decoded frames cap per upload
ALLOWED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

FrameScorer = Callable[[torch.Tensor], torch.Tensor]  # (batch, 1, H, W) -> (batch,) scores


class FrameScoreOut(BaseModel):
    frame_index: int
    score: float
    is_anomalous: bool
    time_s: Optional[float] = None
    source_frame: Optional[int] = None


class IncidentResult(BaseModel):
    incident_id: str
    created_at: str
    source_filename: str
    frame_scores: List[FrameScoreOut]
    max_score: float
    duration_s: Optional[float] = None
    sample_period_s: Optional[float] = None
    truncated: bool = False


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
    """Lazily loads the checkpoint on first use, so the module stays importable (and tests
    run) without a checkpoint on disk."""
    state: dict[str, ConvAutoencoder] = {}

    def score(frames: torch.Tensor) -> torch.Tensor:
        if "model" not in state:
            state["model"] = load_checkpoint(CHECKPOINT_PATH)
        return reconstruction_error(state["model"], frames)

    return score


class SampledVideo:
    def __init__(self, frames: List[torch.Tensor], source_frames: List[int], fps: float,
                 total_read: int, truncated: bool) -> None:
        self.frames = frames
        self.source_frames = source_frames
        self.fps = fps
        self.total_read = total_read
        self.truncated = truncated


def _sample_video(path: Path, every_nth: int = EVERY_NTH_FRAME, max_raw: int = MAX_RAW_FRAMES) -> SampledVideo:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not fps or fps != fps or fps < 1 or fps > 240:
        fps = 25.0
    frames: List[torch.Tensor] = []
    src: List[int] = []
    index = 0
    truncated = False
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if index >= max_raw:
                truncated = True
                break
            if index % every_nth == 0:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, FRAME_SIZE[::-1])
                tensor = torch.from_numpy(resized.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0)
                frames.append(tensor)
                src.append(index)
            index += 1
    finally:
        cap.release()
    return SampledVideo(frames, src, fps, index, truncated)


def _sample_frames_from_video(path: Path, every_nth: int = EVERY_NTH_FRAME) -> List[torch.Tensor]:
    return _sample_video(path, every_nth).frames


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
    cols = {r[1] for r in conn.execute("PRAGMA table_info(incidents)")}
    if "owner" not in cols:
        conn.execute("ALTER TABLE incidents ADD COLUMN owner TEXT NOT NULL DEFAULT ''")
    return conn


def _clean_owner(value: Optional[str]) -> str:
    return value if value and re.fullmatch(r"[A-Za-z0-9-]{8,64}", value) else ""


def _clean_filename(name: Optional[str]) -> str:
    base = Path((name or "unknown").replace("\\", "/")).name
    return re.sub(r"[^\w.\- ()]", "_", base)[:120] or "unknown"


def create_app(
    scorer_factory: Callable[[], FrameScorer] = _default_scorer,
    db_path: Path = DB_PATH,
    samples_dir: Optional[Path] = None,
    max_upload_mb: Optional[float] = None,
) -> FastAPI:
    """Builds the API with an injectable scorer/db path, so tests don't need a trained
    checkpoint on disk."""
    app = FastAPI(title="Incident Intelligence", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("DASHBOARD_ORIGIN", "http://localhost:3000").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    scorer = scorer_factory()
    samples_path = samples_dir if samples_dir is not None else SAMPLES_DIR
    limit_bytes = int((max_upload_mb if max_upload_mb is not None else MAX_UPLOAD_MB) * 1024 * 1024)

    def _score_path(path: Path, display_name: str, owner: str, mode: str, threshold: float, budget: float) -> IncidentResult:
        sv = _sample_video(path)
        if not sv.frames:
            raise HTTPException(status_code=400, detail="no frames could be read from the uploaded video")
        batch = torch.cat(sv.frames, dim=0)
        # Score in small batches to bound memory; scorers accept any batch size.
        scores: List[float] = []
        for i in range(0, batch.shape[0], 64):
            scores.extend(float(s) for s in scorer(batch[i : i + 64]))
        flags, _ = analysis.flag_frames(scores, mode, threshold, budget)
        period = EVERY_NTH_FRAME / sv.fps
        result = IncidentResult(
            incident_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            source_filename=display_name,
            frame_scores=[
                FrameScoreOut(
                    frame_index=i,
                    score=s,
                    is_anomalous=f,
                    time_s=sv.source_frames[i] / sv.fps,
                    source_frame=sv.source_frames[i],
                )
                for i, (s, f) in enumerate(zip(scores, flags))
            ],
            max_score=max(scores),
            duration_s=sv.total_read / sv.fps,
            sample_period_s=period,
            truncated=sv.truncated,
        )
        conn = _get_connection(db_path)
        with conn:
            conn.execute(
                "INSERT INTO incidents (id, created_at, source_filename, result_json, owner) VALUES (?, ?, ?, ?, ?)",
                (result.incident_id, result.created_at, result.source_filename, result.model_dump_json(), owner),
            )
        conn.close()
        return result

    def _load(incident_id: str, owner: str) -> IncidentResult:
        conn = _get_connection(db_path)
        row = conn.execute("SELECT result_json, owner FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        conn.close()
        # Ids are unguessable UUIDs, so an id acts as a share link; the owner column only
        # scopes the history list and stats.
        if row is None:
            raise HTTPException(status_code=404, detail="incident not found")
        return IncidentResult.model_validate_json(row[0])

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model_loaded": CHECKPOINT_PATH.exists()}

    @app.get("/config")
    async def config() -> dict:
        return {
            "max_upload_mb": limit_bytes / 1024 / 1024,
            "allowed_extensions": sorted(ALLOWED_SUFFIXES),
            "absolute_threshold": ANOMALY_THRESHOLD,
            "default_budget": analysis.DEFAULT_BUDGET,
            "every_nth_frame": EVERY_NTH_FRAME,
            "max_raw_frames": MAX_RAW_FRAMES,
        }

    @app.post("/analyze", response_model=IncidentResult)
    async def analyze(
        file: UploadFile = File(...),
        mode: str = Query("absolute"),
        threshold: float = Query(ANOMALY_THRESHOLD, gt=0),
        budget: float = Query(analysis.DEFAULT_BUDGET, gt=0, le=1),
        x_client_id: Optional[str] = Header(None),
    ) -> IncidentResult:
        if mode not in analysis.MODES:
            raise HTTPException(status_code=422, detail=f"mode must be one of {list(analysis.MODES)}")
        name = _clean_filename(file.filename)
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail=f"unsupported file type '{suffix or 'none'}'; use one of {sorted(ALLOWED_SUFFIXES)}",
            )
        # Stream to disk in chunks and stop as soon as the limit is exceeded.
        size = 0
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit_bytes:
                    tmp.close()
                    tmp_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"file is larger than the {limit_bytes // 1024 // 1024} MB upload limit",
                    )
                tmp.write(chunk)
        try:
            return _score_path(tmp_path, name, _clean_owner(x_client_id), mode, threshold, budget)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _sample_file(name: str) -> Path:
        if not re.fullmatch(r"[\w\-]+", name):
            raise HTTPException(status_code=404, detail="sample not found")
        for suffix in ALLOWED_SUFFIXES:
            p = samples_path / f"{name}{suffix}"
            if p.is_file():
                return p
        raise HTTPException(status_code=404, detail="sample not found")

    @app.get("/samples")
    async def list_samples() -> list[dict]:
        if not samples_path.is_dir():
            return []
        out = []
        for p in sorted(samples_path.iterdir()):
            if p.suffix.lower() in ALLOWED_SUFFIXES:
                desc_file = p.with_suffix(".txt")
                desc = desc_file.read_text(encoding="utf-8").strip() if desc_file.exists() else ""
                out.append({"name": p.stem, "filename": p.name, "description": desc,
                            "size_kb": round(p.stat().st_size / 1024)})
        return out

    @app.get("/samples/{name}/video")
    async def sample_video(name: str) -> FileResponse:
        return FileResponse(_sample_file(name))

    @app.post("/samples/{name}/analyze", response_model=IncidentResult)
    async def analyze_sample(
        name: str,
        mode: str = Query("budget"),
        threshold: float = Query(ANOMALY_THRESHOLD, gt=0),
        budget: float = Query(analysis.DEFAULT_BUDGET, gt=0, le=1),
        x_client_id: Optional[str] = Header(None),
    ) -> IncidentResult:
        if mode not in analysis.MODES:
            raise HTTPException(status_code=422, detail=f"mode must be one of {list(analysis.MODES)}")
        p = _sample_file(name)
        return _score_path(p, p.name, _clean_owner(x_client_id), mode, threshold, budget)

    @app.get("/incidents/{incident_id}", response_model=IncidentResult)
    async def get_incident(incident_id: str) -> IncidentResult:
        return _load(incident_id, "")

    @app.delete("/incidents/{incident_id}", status_code=204)
    async def delete_incident(incident_id: str, x_client_id: Optional[str] = Header(None)) -> Response:
        owner = _clean_owner(x_client_id)
        conn = _get_connection(db_path)
        with conn:
            cur = conn.execute("DELETE FROM incidents WHERE id = ? AND owner = ?", (incident_id, owner))
        conn.close()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="incident not found")
        return Response(status_code=204)

    @app.get("/incidents/{incident_id}/report")
    async def incident_report(
        incident_id: str,
        format: str = Query("json", pattern="^(json|csv|md)$"),
        mode: str = Query("budget"),
        threshold: float = Query(ANOMALY_THRESHOLD, gt=0),
        budget: float = Query(analysis.DEFAULT_BUDGET, gt=0, le=1),
        min_z: float = Query(analysis.DEFAULT_MIN_Z, ge=0),
        download: bool = False,
    ) -> Response:
        if mode not in analysis.MODES:
            raise HTTPException(status_code=422, detail=f"mode must be one of {list(analysis.MODES)}")
        result = _load(incident_id, "")
        report = analysis.build_report(result.model_dump(), mode, threshold, budget, min_z)
        if format == "json":
            body, media, ext = analysis.report_json(report), "application/json", "json"
        elif format == "csv":
            body, media, ext = analysis.report_csv(report), "text/csv", "csv"
        else:
            body, media, ext = analysis.report_markdown(report), "text/markdown", "md"
        headers = {}
        if download:
            headers["Content-Disposition"] = f'attachment; filename="incident-report-{incident_id[:8]}.{ext}"'
        return PlainTextResponse(body, media_type=media, headers=headers)

    def _owner_rows(owner: str) -> list[IncidentResult]:
        conn = _get_connection(db_path)
        rows = conn.execute(
            "SELECT result_json FROM incidents WHERE owner = ? ORDER BY created_at DESC", (owner,)
        ).fetchall()
        conn.close()
        return [IncidentResult.model_validate_json(r[0]) for r in rows]

    @app.get("/incidents", response_model=List[IncidentSummary])
    async def list_incidents(limit: int = Query(50, ge=1, le=500), x_client_id: Optional[str] = Header(None)) -> List[IncidentSummary]:
        results = _owner_rows(_clean_owner(x_client_id))[:limit]
        return [
            IncidentSummary(
                incident_id=r.incident_id,
                created_at=r.created_at,
                source_filename=r.source_filename,
                max_score=r.max_score,
                is_anomalous=any(fs.is_anomalous for fs in r.frame_scores),
                frame_count=len(r.frame_scores),
            )
            for r in results
        ]

    @app.get("/stats", response_model=DashboardStats)
    async def get_stats(x_client_id: Optional[str] = Header(None)) -> DashboardStats:
        results = _owner_rows(_clean_owner(x_client_id))
        if not results:
            return DashboardStats(
                total_incidents=0,
                total_frames_analyzed=0,
                anomalous_incident_count=0,
                anomaly_rate=0.0,
                average_max_score=0.0,
            )
        anomalous_count = sum(1 for r in results if any(fs.is_anomalous for fs in r.frame_scores))
        return DashboardStats(
            total_incidents=len(results),
            total_frames_analyzed=sum(len(r.frame_scores) for r in results),
            anomalous_incident_count=anomalous_count,
            anomaly_rate=anomalous_count / len(results),
            average_max_score=sum(r.max_score for r in results) / len(results),
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
