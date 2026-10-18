from __future__ import annotations

import hashlib
import hmac
import os
import re

from app.settings import get_settings

try:
    import phonenumbers as _phonenumbers  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _phonenumbers = None

DEFAULT_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/wismo",
)
PHONE_HMAC_SECRET = os.getenv("PHONE_HMAC_SECRET", "wismo-dev-hmac-secret-key")

_STRIP_CHARS_RE = re.compile(r"[\s\-\(\)\.]")
_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")


def _fallback_normalize_e164(raw: str, default_region: str = "IN") -> str:
    cleaned = _STRIP_CHARS_RE.sub("", raw)
    if cleaned.startswith("00"):
        cleaned = "+" + cleaned[2:]
    if cleaned.startswith("+"):
        if not _E164_RE.match(cleaned):
            raise ValueError(f"Invalid phone number: {raw!r}")
        return cleaned

    region = default_region.upper()
    if region == "IN":
        if re.match(r"^[6-9]\d{9}$", cleaned):
            return f"+91{cleaned}"
        if re.match(r"^0[6-9]\d{9}$", cleaned):
            return f"+91{cleaned[1:]}"
        if re.match(r"^91[6-9]\d{9}$", cleaned):
            return f"+{cleaned}"
    elif region == "US":
        if re.match(r"^[2-9]\d{9}$", cleaned):
            return f"+1{cleaned}"
        if re.match(r"^1[2-9]\d{9}$", cleaned):
            return f"+{cleaned}"

    raise ValueError(f"Invalid phone number: {raw!r}")


def normalize_phone_e164(phone: str, default_region: str = "IN") -> str:
    """Parse and normalize a phone number to E.164 using phonenumbers (default region IN)."""
    raw = phone.strip()
    if not raw:
        raise ValueError("Phone number cannot be empty")
    if _phonenumbers is not None:
        try:
            parsed = _phonenumbers.parse(raw, default_region)
        except _phonenumbers.NumberParseException as exc:
            raise ValueError(f"Invalid phone number: {phone!r}") from exc
        if not _phonenumbers.is_valid_number(parsed):
            raise ValueError(f"Invalid phone number: {phone!r}")
        return _phonenumbers.format_number(parsed, _phonenumbers.PhoneNumberFormat.E164)
    return _fallback_normalize_e164(raw, default_region=default_region)


def hash_phone_e164(e164_phone: str, secret: str | None = None) -> str:
    """Compute HMAC-SHA256 hex digest of an E.164 phone number."""
    normalized = e164_phone.strip()
    if not normalized.startswith("+"):
        raise ValueError(f"Phone number must be in E.164 format starting with '+': {e164_phone!r}")
    resolved_secret = secret if secret is not None else get_settings().phone_hmac_secret
    key = resolved_secret.encode("utf-8")
    return hmac.new(key, normalized.encode("utf-8"), hashlib.sha256).hexdigest()


def normalize_and_hash_phone(
    phone: str,
    *,
    default_region: str = "IN",
    secret: str | None = None,
) -> str:
    """Normalize a raw phone number to E.164 (default region IN) and return its HMAC-SHA256."""
    e164 = normalize_phone_e164(phone, default_region=default_region)
    return hash_phone_e164(e164, secret=secret)


def normalize_email(email: str) -> str:
    """Normalize email to stripped lowercase."""
    cleaned = email.strip().lower()
    if not cleaned:
        raise ValueError("Email cannot be empty")
    return cleaned
