"""
Minimal Supabase PostgREST client for server-side database writes.

Uses the same raw-HTTP approach as supabase_storage.py — deliberately NOT
supabase-py, whose dependency stack conflicts with this backend's pinned
pydantic version.
"""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from app.core.config import settings

log = logging.getLogger("app.services.supabase_db")

_TIMEOUT = 10.0


def _headers() -> Dict[str, str]:
    key = settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_CLIENT_KEY
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def get_match_assets_url(match_id: str) -> Optional[Dict[str, Any]]:
    """Fetch the current assets_url JSONB for a match row (None if absent)."""
    url = (
        f"{settings.SUPABASE_CLIENT_URL}/rest/v1/matches"
        f"?id=eq.{quote(match_id)}&select=assets_url"
    )
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(url, headers=_headers())
    if resp.status_code != 200:
        log.error("matches select failed: %d %s", resp.status_code, resp.text[:200])
        return None
    rows: List[Dict[str, Any]] = resp.json()
    if not rows:
        log.warning("Match %s not found for assets_url read", match_id)
        return None
    return rows[0].get("assets_url") or {}


def update_match_assets_url(match_id: str, assets: Dict[str, Any]) -> bool:
    """Overwrite the assets_url JSONB column for a match row."""
    url = f"{settings.SUPABASE_CLIENT_URL}/rest/v1/matches?id=eq.{quote(match_id)}"
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.patch(
            url,
            headers={**_headers(), "Prefer": "return=minimal"},
            json={"assets_url": assets},
        )
    if resp.status_code not in (200, 204):
        log.error("matches update failed: %d %s", resp.status_code, resp.text[:200])
        return False
    return True


def merge_match_pipeline_assets(match_id: str, pipeline_assets: Dict[str, Any]) -> bool:
    """Read-merge-write: preserves existing sections (e.g. raw) and sets pipeline."""
    existing = get_match_assets_url(match_id)
    if existing is None:
        return False
    merged = {**existing, "pipeline": pipeline_assets}
    return update_match_assets_url(match_id, merged)
