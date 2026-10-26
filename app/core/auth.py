"""
JWT Authentication Dependency

Verifies the Supabase access_token sent in `Authorization: Bearer <token>`.

Verification strategy — the first method that can decide wins:
  1. Local HS256 verification with SUPABASE_JWT_SECRET (if configured). No network.
  2. Local asymmetric verification against the project's JWKS endpoint
     ({SUPABASE_CLIENT_URL}/auth/v1/jwks). Keys are fetched once and cached.
  3. Remote introspection via GET /auth/v1/user (per-request network call).
     Successful verifications are cached for AUTH_CACHE_TTL_SECONDS.

IMPORTANT: the anon key, the publishable key, and the service-role key are JWTs
signed with the SAME secret as user session tokens. A valid signature alone
therefore proves nothing — tokens are only accepted when their role/audience
identifies them as a user session (`role == "authenticated"` or
`aud == "authenticated"`).

Usage in an endpoint:
    from app.core.auth import require_auth, AuthenticatedUser

    @router.post("/something")
    async def my_endpoint(user: AuthenticatedUser = Depends(require_auth)):
        user.user_id   # verified Supabase auth.users UUID
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import httpx
import jwt as pyjwt
from jwt import PyJWKClient
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

log = logging.getLogger("app.core.auth")

# Extracts the Bearer token from the Authorization header.
# auto_error=False lets us return a cleaner 401 message ourselves.
_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class AuthenticatedUser:
    user_id: str       # Supabase auth.users UUID
    email: Optional[str] = None
    role: Optional[str] = None


class _InvalidToken(Exception):
    """The token is definitively invalid (bad claims / expired). Do not fall through."""


# ─── Verification result cache ────────────────────────────────────────────────
# Maps sha256(token) -> (expires_at_epoch, user). Prevents a network round-trip
# for every request when only remote introspection is available.
_auth_cache: Dict[str, Tuple[float, AuthenticatedUser]] = {}
_AUTH_CACHE_MAX_ENTRIES = 10_000


def _cache_get(token_hash: str) -> Optional[AuthenticatedUser]:
    entry = _auth_cache.get(token_hash)
    if not entry:
        return None
    expires_at, user = entry
    if expires_at < time.time():
        _auth_cache.pop(token_hash, None)
        return None
    return user


def _cache_put(token: str, user: AuthenticatedUser, ttl_seconds: Optional[int] = None):
    if len(_auth_cache) > _AUTH_CACHE_MAX_ENTRIES:
        now = time.time()
        for k in [k for k, (exp, _) in _auth_cache.items() if exp < now]:
            _auth_cache.pop(k, None)
        if len(_auth_cache) > _AUTH_CACHE_MAX_ENTRIES:
            _auth_cache.clear()
    ttl = min(ttl_seconds or settings.AUTH_CACHE_TTL_SECONDS, settings.AUTH_CACHE_TTL_SECONDS)
    _auth_cache[hashlib.sha256(token.encode()).hexdigest()] = (time.time() + ttl, user)


# ─── Local verification (runs in a worker thread) ─────────────────────────────

def _user_from_claims(claims: dict) -> AuthenticatedUser:
    """Validate decoded claims and build the user. Rejects non-session JWTs."""
    role = claims.get("role")
    aud = claims.get("aud")
    if role != "authenticated" and aud != "authenticated":
        # anon / publishable / service-role keys are signed with the same secret
        # as user tokens — a valid signature on them must NOT grant access.
        raise _InvalidToken("Token is not a user session token.")
    sub = claims.get("sub")
    if not sub:
        raise _InvalidToken("Token has no subject.")
    return AuthenticatedUser(
        user_id=str(sub),
        email=claims.get("email"),
        role=role or "authenticated",
    )


def _try_hs256(token: str) -> Optional[AuthenticatedUser]:
    if not settings.SUPABASE_JWT_SECRET:
        return None
    try:
        claims = pyjwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False, "require": ["exp", "sub"]},
        )
        return _user_from_claims(claims)
    except pyjwt.ExpiredSignatureError:
        raise _InvalidToken("Token has expired.")
    except _InvalidToken:
        raise
    except pyjwt.InvalidTokenError:
        # Signed by something other than the configured secret (e.g. the
        # project may use asymmetric keys) — let the next strategy decide.
        return None


_jwk_client: Optional[PyJWKClient] = None


def _try_jwks(token: str) -> Optional[AuthenticatedUser]:
    global _jwk_client
    if not settings.SUPABASE_CLIENT_URL:
        return None
    try:
        if _jwk_client is None:
            _jwk_client = PyJWKClient(
                f"{settings.SUPABASE_CLIENT_URL}/auth/v1/jwks",
                cache_keys=True,
                lifespan=300,
            )
        signing_key = _jwk_client.get_signing_key_from_jwt(token)
        claims = pyjwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            options={"verify_aud": False, "require": ["exp", "sub"]},
        )
        return _user_from_claims(claims)
    except pyjwt.ExpiredSignatureError:
        raise _InvalidToken("Token has expired.")
    except _InvalidToken:
        raise
    except Exception as e:  # JWKS unreachable, unknown kid, signature mismatch
        log.debug("JWKS verification unavailable: %s", e)
        return None


def _verify_locally(token: str) -> Optional[AuthenticatedUser]:
    """Try local strategies in order. None means 'undetermined, try remote'."""
    for strategy in (_try_hs256, _try_jwks):
        user = strategy(token)
        if user is not None:
            return user
    return None


# ─── Remote verification (fallback / ground truth) ────────────────────────────

async def _verify_remote(token: str) -> AuthenticatedUser:
    if not settings.SUPABASE_CLIENT_URL or not settings.SUPABASE_CLIENT_KEY:
        log.error("Supabase configuration is missing — cannot verify token")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on this server.",
        )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.SUPABASE_CLIENT_URL}/auth/v1/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "apikey": settings.SUPABASE_CLIENT_KEY,
                },
                timeout=10.0,
            )
    except httpx.TimeoutException:
        log.error("Timeout verifying token with Supabase")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Token verification timed out.",
        )
    except httpx.RequestError as e:
        log.error("Network error verifying token: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Token verification failed.",
        )

    if response.status_code in (401, 403):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if response.status_code != 200:
        log.error("Supabase auth failed: %d %s", response.status_code, response.text)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Token verification service unavailable.",
        )

    user_data = response.json()
    user_id = user_data.get("id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token — missing user ID.",
        )
    return AuthenticatedUser(user_id=user_id, email=user_data.get("email"), role="authenticated")


# ─── FastAPI dependency ───────────────────────────────────────────────────────

def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> AuthenticatedUser:
    """
    FastAPI dependency that verifies the Supabase JWT.

    Raises HTTP 401 when the header is missing or the token is invalid;
    HTTP 503 when the token could not be verified due to server-side issues.
    """
    if not credentials or not credentials.credentials:
        raise _unauthorized(
            "Missing Authorization header. Send: Authorization: Bearer <supabase_access_token>"
        )

    token = credentials.credentials
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    cached = _cache_get(token_hash)
    if cached is not None:
        return cached

    # Local verification does JWT crypto (and possibly a one-time JWKS fetch)
    # — keep it off the event loop.
    try:
        local_user = await asyncio.to_thread(_verify_locally, token)
    except _InvalidToken as e:
        raise _unauthorized(str(e))
    except Exception as e:  # never let crypto errors 500 the request
        log.exception("Local token verification crashed: %s", e)
        local_user = None

    if local_user is not None:
        _cache_put(token, local_user)
        return local_user

    user = await _verify_remote(token)
    _cache_put(token, user)
    return user
