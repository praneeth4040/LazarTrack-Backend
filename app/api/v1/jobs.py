"""
Asynchronous Jobs API

Long pipelines run as background jobs so the client can fire a request and
leave. The client polls (or re-checks on app focus) until the job reaches a
terminal state, then reads `result`.

    POST /api/v1/jobs/match-stats   {match_id, screenshot_url}      -> 202 {job}
    POST /api/v1/jobs/battle-log    {match_id, video_url, run_step7?} -> 202 {job}
    GET  /api/v1/jobs?match_id=…&status=…&include_result=true       -> [jobs]
    GET  /api/v1/jobs/{job_id}                                      -> job
    DELETE /api/v1/jobs/{job_id}                                    -> cancel/delete

All endpoints require authentication; users only ever see their own jobs.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.auth import AuthenticatedUser, require_auth
from app.services.jobs import job_manager

log = logging.getLogger("app.api.v1.jobs")
router = APIRouter(prefix="/jobs", tags=["Jobs"])


# ─── Handlers (bridge job records to the shared pipeline runners) ─────────────

async def _run_match_stats_job(job: dict, progress) -> dict:
    from app.api.v1.extract import run_match_stats_extraction

    return await run_match_stats_extraction(
        match_id=job["payload"]["match_id"],
        screenshot_url=job["payload"]["screenshot_url"],
        progress=progress,
    )


async def _run_battle_log_job(job: dict, progress) -> dict:
    from app.api.v1.battle_log import BattleLogRequest, run_battle_log_extraction

    body = BattleLogRequest(**job["payload"])
    return await run_battle_log_extraction(
        user_id=job["owner_id"],
        body=body,
        progress=progress,
    )


# Concurrency: battle-log is heavy (multi-minute, subprocess pipeline) — one at
# a time; stats extraction is mostly one OpenRouter call — two in parallel.
job_manager.register("match_stats", _run_match_stats_job, concurrency=2, timeout_seconds=600)
job_manager.register("battle_log", _run_battle_log_job, concurrency=1, timeout_seconds=2400)


# ─── Request models ───────────────────────────────────────────────────────────

class MatchStatsJobRequest(BaseModel):
    match_id:       str
    screenshot_url: str          # public URL already in match-assets bucket


class BattleLogJobRequest(BaseModel):
    match_id:  str
    video_url: str               # public URL of the raw video already in match-assets bucket
    run_step7: bool = False


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _public_job(job: dict, include_result: bool = True) -> dict:
    out = {k: v for k, v in job.items() if k != "payload"}
    if not include_result:
        out.pop("result", None)
    return out


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/match-stats", status_code=status.HTTP_202_ACCEPTED, summary="Submit match stats extraction job")
async def submit_match_stats_job(
    body: MatchStatsJobRequest,
    user: AuthenticatedUser = Depends(require_auth),
):
    """Fire-and-forget screenshot stats extraction. Poll GET /jobs/{id} for progress."""
    job = job_manager.create(
        kind="match_stats",
        owner_id=user.user_id,
        match_id=body.match_id,
        payload={"match_id": body.match_id, "screenshot_url": body.screenshot_url},
    )
    return JSONResponse(status_code=202, content=_public_job(job))


@router.post("/battle-log", status_code=status.HTTP_202_ACCEPTED, summary="Submit battle log extraction job")
async def submit_battle_log_job(
    body: BattleLogJobRequest,
    user: AuthenticatedUser = Depends(require_auth),
):
    """Fire-and-forget battle log video extraction (5–10 min). Poll GET /jobs/{id}."""
    job = job_manager.create(
        kind="battle_log",
        owner_id=user.user_id,
        match_id=body.match_id,
        payload={"match_id": body.match_id, "video_url": body.video_url, "run_step7": body.run_step7},
    )
    return JSONResponse(status_code=202, content=_public_job(job))


@router.get("", summary="List my jobs")
async def list_jobs(
    user: AuthenticatedUser = Depends(require_auth),
    match_id: Optional[str] = Query(None, description="Filter by match"),
    kind: Optional[str] = Query(None, description="match_stats | battle_log"),
    job_status: Optional[str] = Query(None, alias="status", description="queued | running | completed | failed | cancelled"),
    include_result: bool = Query(False, description="Include full result payloads (large)"),
):
    jobs = job_manager.list_jobs(
        owner_id=user.user_id,
        match_id=match_id,
        kind=kind,
        status=job_status,
    )
    return {
        "count": len(jobs),
        "jobs": [_public_job(j, include_result=include_result) for j in jobs],
    }


@router.get("/{job_id}", summary="Get job status / progress / result")
async def get_job(
    job_id: str,
    user: AuthenticatedUser = Depends(require_auth),
):
    job = job_manager.get(job_id)
    if job is None or job.get("owner_id") != user.user_id:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _public_job(job)


@router.delete("/{job_id}", summary="Cancel (queued) or delete (finished) a job")
async def cancel_job(
    job_id: str,
    user: AuthenticatedUser = Depends(require_auth),
):
    job = job_manager.get(job_id)
    if job is None or job.get("owner_id") != user.user_id:
        raise HTTPException(status_code=404, detail="Job not found.")
    outcome = job_manager.cancel(job_id)
    if outcome == "missing":
        raise HTTPException(status_code=404, detail="Job not found.")
    if outcome == "running":
        raise HTTPException(
            status_code=409,
            detail="Job is already running and cannot be cancelled. It will finish on its own.",
        )
    return {"message": f"Job {outcome}."}
