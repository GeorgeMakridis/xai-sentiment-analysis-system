"""
Keycloak OIDC authentication for the XAI Platform.

Two modes
---------
- **keycloak** (production): full OIDC login via Keycloak, JWT validation for APIs
- **legacy** (dev fallback): dict-based username/password, Flask session

Set  AUTH_MODE=keycloak  to enable Keycloak.  When it is not set (or set to
``legacy``), the old hardcoded USERS dict is used so devs can run the system
without a Keycloak instance.

Environment Variables (Keycloak mode)
-------------------------------------
AUTH_MODE            = keycloak | legacy  (default: legacy)
KC_SERVER_URL        = http://keycloak:8080   (internal Docker URL)
KC_REALM             = xai-platform
KC_CLIENT_ID         = xai-dashboard
KC_CLIENT_SECRET     = <from Keycloak admin console>
KC_EXTERNAL_URL      = http://localhost:8080  (browser-reachable URL)
PUBLIC_BASE_URL      = optional https://dashboard.example.com  (OAuth redirect_uri; see below)
PREFERRED_URL_SCHEME = set on Flask app in dashboard (http | https); works with ProxyFix

Behind HTTPS, the dashboard uses werkzeug ProxyFix plus PREFERRED_URL_SCHEME so
url_for(..., _external=True) sees the public scheme. If the proxy does not send
trusted X-Forwarded-* headers, set PUBLIC_BASE_URL to the dashboard origin.

Roles
-----
- viewer   – browse datasets, view plots, use chatbot
- analyst  – upload data, run analysis, generate plots
- admin    – manage users, clear data, access all user results
"""

import functools
import logging
import os
from typing import Callable, Dict, List, Optional

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    request,
    session,
    url_for,
)

logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

AUTH_MODE = os.environ.get("AUTH_MODE", "legacy").lower()

KC_SERVER_URL = os.environ.get("KC_SERVER_URL", "http://keycloak:8080")
KC_EXTERNAL_URL = os.environ.get("KC_EXTERNAL_URL", "http://localhost:8080")
KC_REALM = os.environ.get("KC_REALM", "xai-platform")
KC_CLIENT_ID = os.environ.get("KC_CLIENT_ID", "xai-dashboard")
KC_CLIENT_SECRET = os.environ.get("KC_CLIENT_SECRET", "")

# JWKS (public keys) cache
_jwks_client = None
_oauth = None

# Legacy fallback users  — only used when AUTH_MODE=legacy
LEGACY_USERS: Dict[str, str] = {
    "admin": os.environ.get("LEGACY_ADMIN_PW", "changeme"),
    "analyst": os.environ.get("LEGACY_ANALYST_PW", "changeme"),
}

# ─── Keycloak OIDC URLs ─────────────────────────────────────────────────────

def _realm_url(external: bool = False) -> str:
    base = KC_EXTERNAL_URL if external else KC_SERVER_URL
    return f"{base}/realms/{KC_REALM}"


def _oidc_config_url(external: bool = False) -> str:
    return f"{_realm_url(external)}/.well-known/openid-configuration"


def _certs_url() -> str:
    return f"{_realm_url()}/protocol/openid-connect/certs"


def _oauth_redirect_url(endpoint: str) -> str:
    """
    Absolute URL for OIDC redirect_uri and post_logout_redirect_uri.

    If PUBLIC_BASE_URL is set, use it as the origin (when the proxy does not
    send trusted forwarded headers). Otherwise url_for(..., _external=True),
    which respects ProxyFix + PREFERRED_URL_SCHEME on the dashboard app.
    """
    path = url_for(endpoint, _external=False)
    base = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if base:
        return f"{base}{path}"
    return url_for(endpoint, _external=True)


# ─── Initialisation ─────────────────────────────────────────────────────────

def init_auth(app: Flask) -> None:
    """
    Call once at app startup.  Registers OIDC routes if AUTH_MODE=keycloak,
    otherwise registers legacy login/logout routes.
    """
    # Secure the session cookie
    app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)
    app.config.setdefault("SESSION_COOKIE_SAMESITE", "Lax")

    if AUTH_MODE == "keycloak":
        pub = os.environ.get("PUBLIC_BASE_URL", "").strip()
        if pub:
            logger.info("PUBLIC_BASE_URL set for OAuth redirects: %s", pub)
        _init_keycloak(app)
        logger.info("Auth mode: Keycloak OIDC  (realm=%s, client=%s)",
                     KC_REALM, KC_CLIENT_ID)
    else:
        _init_legacy(app)
        logger.info("Auth mode: legacy (dict-based dev accounts)")


# ── Keycloak initialisation ─────────────────────────────────────────────────

def _init_keycloak(app: Flask) -> None:
    """Register OIDC login/callback/logout routes."""
    global _oauth
    try:
        from authlib.integrations.flask_client import OAuth
    except ImportError:
        raise ImportError(
            "authlib is required for Keycloak auth.  "
            "Install with:  pip install authlib"
        )

    _oauth = OAuth(app)
    _oauth.register(
        name="keycloak",
        server_metadata_url=_oidc_config_url(),
        client_id=KC_CLIENT_ID,
        client_secret=KC_CLIENT_SECRET,
        client_kwargs={"scope": "openid email profile"},
    )

    @app.route("/login")
    def kc_login():
        redirect_uri = _oauth_redirect_url("kc_callback")
        return _oauth.keycloak.authorize_redirect(redirect_uri)

    @app.route("/callback")
    def kc_callback():
        token = _oauth.keycloak.authorize_access_token()
        userinfo = token.get("userinfo") or {}

        # Extract roles from realm_access or resource_access
        realm_roles = (
            token.get("access_token_claims", {})
            .get("realm_access", {})
            .get("roles", [])
        )
        # Also try parsing the ID token
        if not realm_roles:
            import jwt as pyjwt
            try:
                claims = pyjwt.decode(
                    token["access_token"],
                    options={"verify_signature": False},
                )
                realm_roles = (
                    claims.get("realm_access", {}).get("roles", [])
                )
            except Exception:
                pass

        session["user_id"] = userinfo.get("preferred_username",
                                          userinfo.get("sub", "unknown"))
        session["email"] = userinfo.get("email", "")
        session["name"] = userinfo.get("name", session["user_id"])
        session["roles"] = realm_roles
        session["access_token"] = token.get("access_token", "")
        session["refresh_token"] = token.get("refresh_token", "")
        session["auth_mode"] = "keycloak"

        return redirect("/")

    @app.route("/logout")
    def kc_logout():
        # Keycloak end-session endpoint
        id_token = session.get("id_token", "")
        session.clear()
        end_session_url = (
            f"{_realm_url(external=True)}/protocol/openid-connect/logout"
            f"?post_logout_redirect_uri={_oauth_redirect_url('kc_login')}"
        )
        if id_token:
            end_session_url += f"&id_token_hint={id_token}"
        return redirect(end_session_url)


# ── Legacy initialisation ───────────────────────────────────────────────────

def _init_legacy(app: Flask) -> None:
    """Register simple form-based login/logout."""

    @app.route("/login", methods=["GET", "POST"])
    def legacy_login():
        from flask import render_template
        if request.method == "POST":
            username = request.form.get("username", "")
            password = request.form.get("password", "")
            if username in LEGACY_USERS and LEGACY_USERS[username] == password:
                session["user_id"] = username
                session["roles"] = ["admin"] if username == "admin" else ["analyst"]
                session["auth_mode"] = "legacy"
                return redirect("/")
            return render_template("login.html", error="Invalid credentials")
        return render_template("login.html")

    @app.route("/logout")
    def legacy_logout():
        session.clear()
        return redirect(url_for("legacy_login"))


# ─── Decorators ──────────────────────────────────────────────────────────────

def login_required(f: Callable) -> Callable:
    """Require an authenticated session (any role)."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"error": "Not authenticated"}), 401
            return redirect(url_for("kc_login" if AUTH_MODE == "keycloak"
                                    else "legacy_login"))
        return f(*args, **kwargs)
    return wrapper


def role_required(*required_roles: str) -> Callable:
    """
    Require the user to have at least one of the specified roles.
    Usage::

        @app.route('/admin-only')
        @role_required('admin')
        def admin_page(): ...
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        @login_required
        def wrapper(*args, **kwargs):
            user_roles = set(session.get("roles", []))
            if not user_roles.intersection(required_roles):
                if request.is_json or request.path.startswith("/api/"):
                    return jsonify({"error": "Forbidden",
                                    "required_roles": list(required_roles)}), 403
                abort(403)
            return f(*args, **kwargs)
        return wrapper
    return decorator


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_current_user() -> Dict[str, str]:
    """Return current user info from the session."""
    return {
        "user_id": session.get("user_id", ""),
        "email": session.get("email", ""),
        "name": session.get("name", ""),
        "roles": session.get("roles", []),
        "auth_mode": session.get("auth_mode", ""),
    }


# ─── Service-to-service JWT validation ──────────────────────────────────────

def validate_service_token(token: str) -> Optional[Dict]:
    """
    Validate a JWT Bearer token for backend-to-backend calls.
    Returns decoded claims on success, None on failure.

    Used by xai_service and ai_outputs to authenticate requests
    forwarded from the dashboard.
    """
    if AUTH_MODE != "keycloak":
        # In legacy mode, trust the user_id header
        return {"preferred_username": "legacy-service", "roles": ["admin"]}

    try:
        from jose import jwt as jose_jwt
        import requests as http

        jwks = http.get(_certs_url(), timeout=5).json()
        claims = jose_jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            audience=KC_CLIENT_ID,
            options={"verify_aud": False},  # audience may differ per client
        )
        return claims
    except ImportError:
        logger.warning("python-jose not installed — JWT validation disabled")
        return None
    except Exception as exc:
        logger.warning("JWT validation failed: %s", exc)
        return None


def get_bearer_token() -> Optional[str]:
    """Extract Bearer token from the Authorization header."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


def service_auth_middleware():
    """
    Flask before_request handler for backend services.
    Validates JWT from Authorization header and sets request.user_id.

    Usage in xai_service/app.py or ai_outputs/app.py::

        from auth import service_auth_middleware
        app.before_request(service_auth_middleware)
    """
    # Health endpoint is always public
    if request.path in ("/health", "/"):
        return None

    if AUTH_MODE != "keycloak":
        # Legacy mode: trust X-User-Id header from dashboard
        request.user_id = request.headers.get(
            "X-User-Id",
            request.json.get("user_id", "anonymous") if request.is_json else "anonymous",
        )
        request.roles = ["admin"]
        return None

    token = get_bearer_token()
    if not token:
        return jsonify({"error": "Missing Authorization header"}), 401

    claims = validate_service_token(token)
    if claims is None:
        return jsonify({"error": "Invalid or expired token"}), 401

    request.user_id = claims.get("preferred_username", "unknown")
    request.roles = claims.get("realm_access", {}).get("roles", [])
    return None
