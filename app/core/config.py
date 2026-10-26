import os
import logging
from dotenv import load_dotenv

_backend_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
if os.path.exists(_backend_env):
    load_dotenv(_backend_env)
else:
    load_dotenv()

class Settings:
    PROJECT_NAME: str = "LazarTrack Intelligent Extraction Agent"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"

    MAX_IMAGE_BYTES: int = 20 * 1024 * 1024  # 20 MB
    MAX_VIDEO_BYTES: int = 500 * 1024 * 1024  # 500 MB

    # Video pipeline configuration
    VIDEO_OUTPUT_DIR: str = os.getenv(
        "VIDEO_OUTPUT_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "video", "output")
    )
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

    # ── Supabase Storage (file uploads from backend) ──────────────────────────
    SUPABASE_CLIENT_URL: str = os.getenv("SUPABASE_CLIENT_URL", "").rstrip("/")
    SUPABASE_CLIENT_KEY: str = os.getenv("SUPABASE_CLIENT_KEY", "")
    SUPABASE_BUCKET: str = "match-assets"

    # Server-side only: service-role key for Storage uploads and database
    # writes (matches.assets_url updates). NEVER expose this to clients.
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

    # ── Supabase JWT verification ─────────────────────────────────────────────
    # The JWT secret is the Supabase project's JWT secret (Settings → API → JWT Secret).
    # Used to verify the access_token sent by the frontend in the Authorization header.
    # If unset (or if the project uses asymmetric signing keys), tokens are verified
    # against the JWKS endpoint, falling back to Supabase's /auth/v1/user API.
    SUPABASE_JWT_SECRET: str = os.getenv("SUPABASE_JWT_SECRET", "")

    # How long a successfully verified token is trusted without re-verification.
    AUTH_CACHE_TTL_SECONDS: int = int(os.getenv("AUTH_CACHE_TTL_SECONDS", "300"))

    # ── CORS (web origins allowed to call this API from a browser) ────────────
    # Comma-separated list. Set to "*" to allow every origin (local testing only).
    ALLOWED_ORIGINS: str = os.getenv(
        "ALLOWED_ORIGINS",
        ",".join([
            "https://lazartrack.app",
            "https://www.lazartrack.app",
            "https://app.lazartrack.app",
            "https://lazarflow.app",
            "https://www.lazarflow.app",
            "http://localhost:8081",
            "http://127.0.0.1:8081",
            "http://localhost:19006",
            "http://localhost:3000",
        ]),
    )
    # Extra regex for dynamic preview deployments (e.g. Vercel previews).
    # Empty string disables regex matching.
    ALLOWED_ORIGIN_REGEX: str = os.getenv(
        "ALLOWED_ORIGIN_REGEX", r"^https://[a-z0-9-]+\.vercel\.app$"
    )

    # ── Outbound asset fetch allowlist ────────────────────────────────────────
    # The extract/battle-log endpoints download user-supplied URLs server-side.
    # Restrict those fetches to these host suffixes to prevent authenticated
    # users from pointing the backend at internal endpoints (SSRF).
    # Comma-separated; empty string disables the restriction (not recommended).
    ALLOWED_ASSET_HOSTS: str = os.getenv("ALLOWED_ASSET_HOSTS", ".supabase.co")

    def validate(self):
        if not self.OPENROUTER_API_KEY:
            logging.warning("OPENROUTER_API_KEY not set — VL extraction will fail")
        if not self.SUPABASE_CLIENT_URL or not self.SUPABASE_CLIENT_KEY:
            logging.warning("SUPABASE_CLIENT_URL/KEY not set — authentication will fail")
        if not self.SUPABASE_JWT_SECRET:
            logging.info(
                "SUPABASE_JWT_SECRET not set — tokens will be verified via JWKS / "
                "Supabase /auth/v1/user"
            )

settings = Settings()
settings.validate()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    )
