"""
Supabase Storage Service for LazarTrack

Handles uploading battle log outputs, zone tracker outputs, and match assets
to Supabase Storage (LazarTrack project) in a structured directory hierarchy:
  userid/matchid/battle_log/<files>
  userid/matchid/zone_tracker/<files>
  userid/matchid/<screenshots>

Uses raw HTTP requests with the service role key (admin credentials) to bypass RLS.
Storage bucket: match-assets (public bucket)

Note: We use the service role key for uploads to bypass RLS policies that would
normally restrict writes from the client-side anon key.
"""

import os
import logging
from typing import Optional
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

# Get Supabase credentials from environment
SUPABASE_URL = os.getenv("SUPABASE_CLIENT_URL", "").rstrip("/")
# For uploads, use service role key to bypass RLS (backend-to-storage communication)
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_CLIENT_KEY", "")
BUCKET_NAME = "match-assets"  # Public bucket in LazarTrack project

if not SUPABASE_URL:
    log.warning("SUPABASE_CLIENT_URL not configured. Storage uploads will fail.")

if not SUPABASE_SERVICE_ROLE_KEY:
    log.warning(
        "SUPABASE_SERVICE_ROLE_KEY not configured. Storage uploads will fail. "
        "Set SUPABASE_SERVICE_ROLE_KEY in .env (found in Supabase Dashboard → Settings → API)"
    )


class SupabaseStorageClient:
    """Client for uploading files to Supabase Storage using raw HTTP requests"""

    def __init__(self):
        self.bucket = BUCKET_NAME
        self.base_url = SUPABASE_URL
        # Use service role key for backend uploads (bypasses RLS)
        self.api_key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY
        self._http_client = None

    @property
    def http_client(self) -> httpx.Client:
        """Get or create HTTP client with service role key for backend uploads"""
        if self._http_client is None:
            self._http_client = httpx.Client(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "apikey": self.api_key,
                }
            )
        return self._http_client

    def upload_file(
        self,
        file_path: str,
        remote_path: str,
        content_type: str = "application/octet-stream",
    ) -> Optional[str]:
        """
        Upload a file to Supabase Storage (LazarTrack project, match-assets bucket)

        Args:
            file_path: Local file path to upload
            remote_path: Path in bucket (e.g., "userid/matchid/battle_log/group_000.png")
            content_type: MIME type of the file

        Returns:
            Public URL of the uploaded file, or None on error
        """
        if not os.path.exists(file_path):
            log.error(f"File not found: {file_path}")
            return None

        if not self.base_url or not self.api_key:
            log.error(
                "Supabase credentials not configured. "
                "Set SUPABASE_CLIENT_URL and SUPABASE_SERVICE_ROLE_KEY in .env"
            )
            return None

        try:
            with open(file_path, "rb") as f:
                file_data = f.read()

            return self._do_upload(remote_path, file_data, content_type)

        except Exception as e:
            log.error(f"Error uploading {remote_path}: {e}", exc_info=True)
            return None

    def upload_file_from_bytes(
        self,
        file_bytes: bytes,
        remote_path: str,
        content_type: str = "application/octet-stream",
    ) -> Optional[str]:
        """
        Upload file content from bytes

        Args:
            file_bytes: File content as bytes
            remote_path: Path in bucket
            content_type: MIME type

        Returns:
            Public URL or None on error
        """
        if not self.base_url or not self.api_key:
            log.error(
                "Supabase credentials not configured. "
                "Set SUPABASE_CLIENT_URL and SUPABASE_SERVICE_ROLE_KEY in .env"
            )
            return None

        try:
            return self._do_upload(remote_path, file_bytes, content_type)
        except Exception as e:
            log.error(f"Error uploading {remote_path}: {e}", exc_info=True)
            return None

    def _do_upload(
        self, remote_path: str, file_data: bytes, content_type: str
    ) -> Optional[str]:
        """Internal: Perform the actual upload via HTTP request"""
        # Construct the upload URL
        upload_url = f"{self.base_url}/storage/v1/object/{self.bucket}/{remote_path}"

        try:
            response = self.http_client.post(
                upload_url,
                content=file_data,
                headers={"Content-Type": content_type},
                params={"upsert": "true"},
            )

            if response.status_code not in (200, 201):
                log.error(
                    f"Upload failed with status {response.status_code}: {response.text}"
                )
                return None

            log.info(f"✓ Uploaded {remote_path} to Supabase Storage")
            # Construct and return the public URL
            return self.get_public_url(remote_path)

        except Exception as e:
            log.error(f"HTTP request error uploading {remote_path}: {e}", exc_info=True)
            return None

    def delete_file(self, remote_path: str) -> bool:
        """Delete a file from Supabase Storage"""
        if not self.base_url or not self.api_key:
            return False

        try:
            delete_url = f"{self.base_url}/storage/v1/object/{self.bucket}/{remote_path}"
            response = self.http_client.delete(delete_url)

            if response.status_code not in (200, 204):
                log.error(
                    f"Delete failed with status {response.status_code}: {response.text}"
                )
                return False

            return True
        except Exception as e:
            log.error(f"Error deleting {remote_path}: {e}")
            return False

    def get_public_url(self, remote_path: str) -> str:
        """Get the public URL for a file"""
        return f"{self.base_url}/storage/v1/object/public/{self.bucket}/{remote_path}"


# Singleton instance
storage_client = SupabaseStorageClient()
