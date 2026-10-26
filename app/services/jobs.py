"""
In-process asynchronous job manager.

Long pipelines (screenshot stats, battle-log video extraction) run as
background jobs instead of blocking the HTTP request:

    POST /api/v1/jobs/match-stats  -> 202 {job}   (user triggers and leaves)
    GET  /api/v1/jobs/{job_id}     -> live status / progress / final result

Design:
  - One asyncio task per job; per-kind semaphores cap concurrency so a burst
    of submissions cannot exhaust the container's CPU/RAM.
  - Every job record is persisted to {VIDEO_OUTPUT_DIR}/jobs/{job_id}.json on
    each transition, so status survives process restarts. On startup, jobs
    left in queued/running by a previous process are marked failed
    (in-process pipelines cannot be resumed).
  - Terminal jobs older than JOB_RETENTION_DAYS are pruned on startup.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.core.config import settings

log = logging.getLogger("app.services.jobs")

JOB_RETENTION_DAYS = 7

# progress(stage, message, percent)
ProgressFn = Callable[[str, str, Optional[float]], None]
# handler(job_record, progress_fn) -> jsonable result dict
JobHandler = Callable[[dict, ProgressFn], Awaitable[dict]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self):
        self._jobs: Dict[str, dict] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._handlers: Dict[str, dict] = {}
        self._jobs_dir = Path(settings.VIDEO_OUTPUT_DIR) / "jobs"

    # ── Registration ────────────────────────────────────────────────────────
    def register(
        self,
        kind: str,
        handler: JobHandler,
        concurrency: int = 2,
        timeout_seconds: int = 1800,
    ) -> None:
        if kind in self._handlers:
            raise RuntimeError(f"Job kind '{kind}' is already registered")
        self._handlers[kind] = {
            "handler": handler,
            "timeout": timeout_seconds,
        }
        self._semaphores[kind] = asyncio.Semaphore(concurrency)

    # ── Persistence ─────────────────────────────────────────────────────────
    def _job_file(self, job_id: str) -> Path:
        return self._jobs_dir / f"{job_id}.json"

    def _persist(self, job: dict) -> None:
        try:
            self._jobs_dir.mkdir(parents=True, exist_ok=True)
            with open(self._job_file(job["id"]), "w") as f:
                json.dump(job, f, indent=2, default=str)
        except Exception as e:
            log.warning("Could not persist job %s: %s", job["id"], e)

    def _load_from_disk(self, job_id: str) -> Optional[dict]:
        f = self._job_file(job_id)
        if not f.exists():
            return None
        try:
            with open(f) as fh:
                return json.load(fh)
        except Exception as e:
            log.warning("Could not read job file %s: %s", f, e)
            return None

    # ── Lifecycle ───────────────────────────────────────────────────────────
    def create(self, kind: str, owner_id: str, payload: dict, match_id: Optional[str] = None) -> dict:
        if kind not in self._handlers:
            raise KeyError(f"Unknown job kind '{kind}'")
        job_id = uuid.uuid4().hex[:16]
        job = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "owner_id": owner_id,
            "match_id": match_id,
            "payload": payload,
            "progress": {
                "stage": "queued",
                "message": "Queued — waiting for a worker…",
                "percent": None,
            },
            "result": None,
            "error": None,
            "created_at": _now_iso(),
            "started_at": None,
            "finished_at": None,
        }
        self._jobs[job_id] = job
        self._persist(job)
        self._tasks[job_id] = asyncio.create_task(self._run(job_id))
        log.info("[jobs] created %s job %s for user=%s match=%s", kind, job_id, owner_id, match_id)
        return dict(job)

    async def _run(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        sem = self._semaphores[job["kind"]]
        await sem.acquire()
        try:
            # The job may have been cancelled while queued.
            if job["status"] == "cancelled":
                return
            job["status"] = "running"
            job["started_at"] = _now_iso()
            self._set_progress(job_id, "starting", "Starting pipeline…", 5)
            self._persist(job)

            spec = self._handlers[job["kind"]]
            progress = lambda stage, message, percent=None: self._set_progress(  # noqa: E731
                job_id, stage, message, percent
            )
            try:
                result = await asyncio.wait_for(spec["handler"](job, progress), spec["timeout"])
                job["result"] = result
                job["status"] = "completed"
                self._set_progress(job_id, "done", "Completed", 100)
                log.info("[jobs] job %s completed", job_id)
            except asyncio.TimeoutError:
                job["status"] = "failed"
                job["error"] = "Job timed out."
                log.warning("[jobs] job %s timed out", job_id)
            except Exception as e:
                detail = getattr(e, "detail", None) or str(e)
                job["status"] = "failed"
                job["error"] = str(detail)
                log.exception("[jobs] job %s failed: %s", job_id, e)
            job["finished_at"] = _now_iso()
            self._persist(job)
        finally:
            sem.release()
            self._tasks.pop(job_id, None)

    def _set_progress(self, job_id: str, stage: str, message: str, percent: Optional[float]) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job["progress"] = {"stage": stage, "message": message, "percent": percent}
        self._persist(job)

    # ── Queries ─────────────────────────────────────────────────────────────
    def get(self, job_id: str) -> Optional[dict]:
        if job_id in self._jobs:
            return dict(self._jobs[job_id])
        job = self._load_from_disk(job_id)
        if job is not None:
            self._jobs[job_id] = job
            return dict(job)
        return None

    def list_jobs(
        self,
        owner_id: str,
        match_id: Optional[str] = None,
        kind: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """List a user's jobs, newest first. Loads the jobs dir on first call."""
        jobs: List[dict] = []
        if self._jobs_dir.exists():
            for f in self._jobs_dir.glob("*.json"):
                if f.stem not in self._jobs:
                    job = self._load_from_disk(f.stem)
                    if job is not None:
                        self._jobs[f.stem] = job
        for job in self._jobs.values():
            if job.get("owner_id") != owner_id:
                continue
            if match_id and job.get("match_id") != match_id:
                continue
            if kind and job.get("kind") != kind:
                continue
            if status and job.get("status") != status:
                continue
            jobs.append(dict(job))
        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
        return jobs[:limit]

    # ── Cancellation / deletion ─────────────────────────────────────────────
    def cancel(self, job_id: str) -> str:
        """Cancel a queued job or delete a terminal one. Returns the action taken."""
        job = self.get(job_id)
        if job is None:
            return "missing"
        status = job["status"]
        if status == "queued":
            job["status"] = "cancelled"
            job["finished_at"] = _now_iso()
            self._persist(job)
            return "cancelled"
        if status in ("completed", "failed", "cancelled"):
            self._jobs.pop(job_id, None)
            try:
                self._job_file(job_id).unlink(missing_ok=True)
            except Exception:
                pass
            return "deleted"
        return "running"  # cannot stop a pipeline mid-flight

    # ── Startup recovery ────────────────────────────────────────────────────
    def recover(self) -> None:
        """Mark jobs orphaned by a restart as failed and prune old terminal jobs."""
        if not self._jobs_dir.exists():
            return
        now = time.time()
        for f in self._jobs_dir.glob("*.json"):
            job = self._load_from_disk(f.name[: -len(".json")])
            if job is None:
                continue
            if job.get("status") in ("queued", "running"):
                job["status"] = "failed"
                job["error"] = "Interrupted by a server restart. Please submit again."
                job["finished_at"] = _now_iso()
                self._persist(job)
                log.warning("[jobs] recovered orphaned job %s -> failed", job.get("id"))
            created = job.get("created_at")
            if job.get("status") in ("completed", "failed", "cancelled"):
                try:
                    age_days = (now - datetime.fromisoformat(created).timestamp()) / 86400 \
                        if created else 0
                    if age_days > JOB_RETENTION_DAYS:
                        f.unlink(missing_ok=True)
                except Exception:
                    pass


job_manager = JobManager()
