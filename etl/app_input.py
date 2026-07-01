"""
etl/app_input.py
==================
Parses and validates a user-supplied Google Play app identifier, which may
be a raw app id (e.g. "com.netflix.mediaclient") or a full Play Store URL
(e.g. "https://play.google.com/store/apps/details?id=com.netflix.mediaclient").
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

_APP_ID_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z][a-zA-Z0-9_]*)+$")


class InvalidAppIdentifierError(ValueError):
    """Raised when user input is neither a valid app id nor a valid Play Store URL."""


def is_valid_app_id(app_id: str) -> bool:
    """Android package names: dot-separated segments, each starting with a letter."""
    return bool(_APP_ID_PATTERN.match(app_id))


def extract_app_id(user_input: str) -> str:
    """
    Accept either a raw Android app id or a Google Play Store URL and return
    a validated app id.

    Raises
    ------
    InvalidAppIdentifierError
        If the input is empty, the URL is not a play.google.com URL, the URL
        has no `id` query parameter, or the resulting app id does not match
        Android package-name format.
    """
    if not user_input or not user_input.strip():
        raise InvalidAppIdentifierError("Input cannot be empty.")

    candidate = user_input.strip()

    if candidate.lower().startswith("http://") or candidate.lower().startswith("https://"):
        parsed = urlparse(candidate)
        if "play.google.com" not in parsed.netloc.lower():
            raise InvalidAppIdentifierError(
                "URL must be a Google Play Store URL (play.google.com)."
            )
        query_params = parse_qs(parsed.query)
        ids = query_params.get("id")
        if not ids or not ids[0]:
            raise InvalidAppIdentifierError(
                "Could not find an 'id' parameter in the Play Store URL."
            )
        candidate = ids[0]

    if not is_valid_app_id(candidate):
        raise InvalidAppIdentifierError(
            f"'{candidate}' is not a valid Android app id "
            "(expected reverse-domain format, e.g. com.example.app)."
        )

    return candidate