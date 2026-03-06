"""
Shared authentication module.
Supports Keycloak OIDC (production) and legacy dict-based auth (dev fallback).
"""
from .keycloak import (
    init_auth,
    login_required,
    role_required,
    get_current_user,
    validate_service_token,
    AUTH_MODE,
)

__all__ = [
    "init_auth",
    "login_required",
    "role_required",
    "get_current_user",
    "validate_service_token",
    "AUTH_MODE",
]
