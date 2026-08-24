"""Background assessment jobs.

An assessment takes seconds to minutes, which is too long for a request cycle
and too short to justify a broker. Jobs run on a small thread pool; status is
polled; results are written to disk so a restart does not lose them and so the
chip endpoints can read rasters back without holding arrays in memory.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .assess import AssessmentResult
from .export import parcels_csv
from .pipeline import AssessmentRequest, PipelineError, run_assessment

log = logging.getLogger(__name__)

QUEUED, RUNNING, DONE, ERROR = "queued", "running", "done", "error"
RESULTS_IN_MEMORY = 8


@dataclass
class Job:
    id: str
    request: AssessmentRequest
    status: str = QUEUED
    progress: float = 0.0
    message: str = "queued"
    created_at: str = field(default_factory=lambda: _now())
    finished_at: str | None = None
    error: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def directory(self) -> Path:
        return Path(config.JOB_DIR) / self.id

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "summary": self.summary,
            "warnings": self.warnings,
            "request": {
                "pre_scene_id": self.request.pre_scene_id,
                "post_scene_id": self.request.post_scene_id,
                "bbox": list(self.request.bbox) if self.request.bbox else None,
                "parcel_count": len(self.request.parcel_ids),
                "gsd": self.request.gsd,
                "building_source": self.request.building_source,
            },
        }


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class JobRegistry:
    """Submit, track and serve assessment jobs."""

    def __init__(self, workers: int = 2):
        self._jobs: dict[str, Job] = {}
        self._results: "OrderedDict[str, AssessmentResult]" = OrderedDict()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="ddx-assess")

    # -- lifecycle -----------------------------------------------------------
    def submit(self, request: AssessmentRequest) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], request=request)
        with self._lock:
            self._jobs[job.id] = job
        job.directory.mkdir(parents=True, exist_ok=True)
        self._write_status(job)
        self._pool.submit(self._run, job)
        return job

    def _run(self, job: Job) -> None:
        def progress(message: str, fraction: float) -> None:
            job.message = message
            job.progress = fraction
            self._write_status(job)

        job.status = RUNNING
        job.message = "starting"
        self._write_status(job)
        try:
            result = run_assessment(job.request, progress=progress,
                                    artifact_dir=job.directory)
            self._persist(job, result)
            job.summary = result.summary
            job.warnings = result.warnings
            job.status = DONE
            job.message = "complete"
            job.progress = 1.0
        except PipelineError as exc:
            job.status, job.error, job.message = ERROR, str(exc), "failed"
            log.info("job %s rejected: %s", job.id, exc)
        except Exception as exc:
            job.status = ERROR
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = "failed"
            log.error("job %s failed\n%s", job.id, traceback.format_exc())
        finally:
            job.finished_at = _now()
            self._write_status(job)

    def _persist(self, job: Job, result: AssessmentResult) -> None:
        with self._lock:
            self._results[job.id] = result
            self._results.move_to_end(job.id)
            while len(self._results) > RESULTS_IN_MEMORY:
                self._results.popitem(last=False)
        directory = job.directory
        (directory / "parcels.geojson").write_text(
            json.dumps(result.feature_collection(geometry=True)))
        (directory / "buildings.geojson").write_text(
            json.dumps(result.building_feature_collection()))
        (directory / "summary.json").write_text(json.dumps(result.summary, indent=2))
        (directory / "parcels.csv").write_text(parcels_csv(result))

    def _write_status(self, job: Job) -> None:
        try:
            (job.directory / "status.json").write_text(json.dumps(job.as_dict(), indent=2))
        except OSError as exc:      # status file is a convenience, not the source of truth
            log.debug("could not write status for %s: %s", job.id, exc)

    # -- access --------------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        # Recover a job from a previous process run.
        status_file = Path(config.JOB_DIR) / job_id / "status.json"
        if not status_file.exists():
            return None
        try:
            raw = json.loads(status_file.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        restored = Job(
            id=raw["id"],
            request=AssessmentRequest(
                pre_scene_id=raw["request"]["pre_scene_id"],
                post_scene_id=raw["request"]["post_scene_id"],
                bbox=tuple(raw["request"]["bbox"]) if raw["request"].get("bbox") else None,
            ),
            status=raw.get("status", DONE), progress=raw.get("progress", 1.0),
            message=raw.get("message", ""), created_at=raw.get("created_at", _now()),
            finished_at=raw.get("finished_at"), error=raw.get("error"),
            summary=raw.get("summary", {}), warnings=raw.get("warnings", []),
        )
        with self._lock:
            self._jobs[job_id] = restored
        return restored

    def result(self, job_id: str) -> AssessmentResult | None:
        with self._lock:
            return self._results.get(job_id)

    def artifact(self, job_id: str, name: str) -> Path | None:
        path = Path(config.JOB_DIR) / job_id / name
        return path if path.exists() else None

    def recent(self, limit: int = 20) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


registry = JobRegistry()
