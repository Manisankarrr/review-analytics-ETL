"""
etl/config.py
==============
Centralized configuration for the ETL pipeline and the FastAPI service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

_REQUIRED_VARS: List[str] = [
    "AIVEN_HOST",
    "AIVEN_PORT",
    "AIVEN_DATABASE",
    "AIVEN_USER",
    "AIVEN_PASSWORD",
    "GOOGLE_PLAY_APP_ID",
]


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    aiven_host: str
    aiven_port: int
    aiven_database: str
    aiven_user: str
    aiven_password: str
    aiven_ssl_ca_path: Optional[str]

    google_play_app_ids: List[str]
    google_play_lang: str
    google_play_country: str
    reviews_per_run: int

    raw_data_path: Path
    processed_reviews_path: Path
    processed_keywords_path: Path
    etl_stats_path: Path

    log_level: str

    @property
    def database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.aiven_user}:{self.aiven_password}"
            f"@{self.aiven_host}:{self.aiven_port}/{self.aiven_database}"
        )

    @property
    def ssl_connect_args(self) -> dict:
        if self.aiven_ssl_ca_path and Path(self.aiven_ssl_ca_path).is_file():
            return {"ssl": {"ca": self.aiven_ssl_ca_path}}
        return {"ssl": {}}


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name, default)
    if value is not None:
        value = value.strip()
    return value


def _validate_required() -> List[str]:
    missing = []
    for var in _REQUIRED_VARS:
        value = os.environ.get(var)
        if value is None or value.strip() == "":
            missing.append(var)
    return missing


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    missing = _validate_required()
    if missing:
        raise ConfigError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in real values, or set "
            "them as GitHub Actions secrets / platform environment variables."
        )

    try:
        aiven_port = int(_get_env("AIVEN_PORT"))
    except (TypeError, ValueError) as exc:
        raise ConfigError("AIVEN_PORT must be an integer (e.g. 12691).") from exc

    app_ids_raw = _get_env("GOOGLE_PLAY_APP_ID", "")
    app_ids = [a.strip() for a in app_ids_raw.split(",") if a.strip()]
    if not app_ids:
        raise ConfigError(
            "GOOGLE_PLAY_APP_ID must contain at least one app id "
            "(comma-separated for multiple apps), e.g. com.spotify.music"
        )

    try:
        reviews_per_run = int(_get_env("REVIEWS_PER_RUN", "200"))
    except ValueError as exc:
        raise ConfigError("REVIEWS_PER_RUN must be an integer.") from exc

    raw_path = _get_env("RAW_DATA_PATH", "data/raw/raw_reviews.csv")
    processed_reviews = _get_env(
        "PROCESSED_REVIEWS_PATH", "data/processed/clean_reviews.csv"
    )
    processed_keywords = _get_env(
        "PROCESSED_KEYWORDS_PATH", "data/processed/keywords.csv"
    )
    stats_path = _get_env("ETL_STATS_PATH", "data/processed/etl_stats.json")

    return Settings(
        aiven_host=_get_env("AIVEN_HOST"),
        aiven_port=aiven_port,
        aiven_database=_get_env("AIVEN_DATABASE"),
        aiven_user=_get_env("AIVEN_USER"),
        aiven_password=_get_env("AIVEN_PASSWORD"),
        aiven_ssl_ca_path=_get_env("AIVEN_SSL_CA_PATH"),
        google_play_app_ids=app_ids,
        google_play_lang=_get_env("GOOGLE_PLAY_LANG", "en"),
        google_play_country=_get_env("GOOGLE_PLAY_COUNTRY", "us"),
        reviews_per_run=reviews_per_run,
        raw_data_path=PROJECT_ROOT / raw_path,
        processed_reviews_path=PROJECT_ROOT / processed_reviews,
        processed_keywords_path=PROJECT_ROOT / processed_keywords,
        etl_stats_path=PROJECT_ROOT / stats_path,
        log_level=_get_env("LOG_LEVEL", "INFO").upper(),
    )