"""
etl/extract.py
===============
Extract stage: pulls reviews for one or more Google Play apps using
google-play-scraper, with pagination, retry-with-backoff on transient
failures, and structured logging of extraction statistics.

Run as a module from the project root:
    python -m etl.extract
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from google_play_scraper import Sort, reviews
from google_play_scraper.exceptions import NotFoundError

from etl.config import Settings, get_settings

logger = logging.getLogger(__name__)

_EXPECTED_FIELDS = [
    "reviewId",
    "userName",
    "content",
    "score",
    "thumbsUpCount",
    "reviewCreatedVersion",
    "at",
    "appVersion",
]

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2
PAGE_SIZE = 200


def _fetch_page_with_retry(
    app_id: str,
    lang: str,
    country: str,
    continuation_token: Optional[Any],
) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
    """
    Fetch a single page of reviews, retrying transient failures with
    exponential backoff. Raises the last exception if all retries fail.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result, token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=PAGE_SIZE,
                continuation_token=continuation_token,
            )
            return result, token
        except NotFoundError as exc:
            logger.error("App id '%s' not found on Google Play: %s", app_id, exc)
            raise
        except Exception as exc:
            last_exc = exc
            wait = BACKOFF_BASE_SECONDS**attempt
            logger.warning(
                "Transient error fetching reviews for %s (attempt %d/%d): %s. "
                "Retrying in %ds.",
                app_id,
                attempt,
                MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)
    logger.error(
        "Exhausted %d retries fetching reviews for %s.", MAX_RETRIES, app_id
    )
    raise last_exc  # type: ignore[misc]


def extract_reviews_for_app(
    app_id: str,
    lang: str,
    country: str,
    target_count: int,
) -> List[Dict[str, Any]]:
    """
    Extract up to `target_count` reviews for a single app, paginating via
    the continuation token until either the target is reached or the API
    reports no further pages.
    """
    collected: List[Dict[str, Any]] = []
    continuation_token = None
    pages_fetched = 0

    while len(collected) < target_count:
        try:
            page_results, continuation_token = _fetch_page_with_retry(
                app_id, lang, country, continuation_token
            )
        except NotFoundError:
            break
        except Exception:
            logger.exception(
                "Giving up on app %s after repeated failures; keeping %d "
                "reviews collected so far.",
                app_id,
                len(collected),
            )
            break

        pages_fetched += 1

        if not page_results:
            logger.info(
                "No more reviews returned for %s after %d page(s).",
                app_id,
                pages_fetched,
            )
            break

        collected.extend(page_results)
        logger.info(
            "App %s: fetched page %d (%d reviews, %d total so far).",
            app_id,
            pages_fetched,
            len(page_results),
            len(collected),
        )

        if continuation_token is None or getattr(continuation_token, "token", None) is None:
            logger.info("Reached end of available reviews for %s.", app_id)
            break

    return collected[:target_count]


def _to_dataframe(raw_reviews: List[Dict[str, Any]], app_id: str) -> pd.DataFrame:
    """Convert a list of raw review dicts into a structured DataFrame."""
    if not raw_reviews:
        return pd.DataFrame(columns=_EXPECTED_FIELDS + ["app_id"])

    df = pd.DataFrame(raw_reviews)

    for field in _EXPECTED_FIELDS:
        if field not in df.columns:
            df[field] = None

    df = df[_EXPECTED_FIELDS].copy()
    df["app_id"] = app_id
    return df


def run_extraction(settings: Settings, app_ids: Optional[List[str]] = None) -> pd.DataFrame:
    
    target_app_ids = app_ids if app_ids is not None else settings.google_play_app_ids
    all_frames: List[pd.DataFrame] = []

    for app_id in target_app_ids:
        logger.info("Starting extraction for app_id=%s", app_id)
        raw_reviews = extract_reviews_for_app(
            app_id=app_id,
            lang=settings.google_play_lang,
            country=settings.google_play_country,
            target_count=settings.reviews_per_run,
        )
        logger.info("Extraction complete for %s: %d reviews.", app_id, len(raw_reviews))
        all_frames.append(_to_dataframe(raw_reviews, app_id))

    combined = (
        pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    )
    return combined


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = get_settings()

    logger.info(
        "Extraction starting for app id(s): %s (target %d reviews/app)",
        ", ".join(settings.google_play_app_ids),
        settings.reviews_per_run,
    )

    df = run_extraction(settings)

    settings.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.raw_data_path, index=False)

    logger.info(
        "Extraction finished. %d total reviews written to %s",
        len(df),
        settings.raw_data_path,
    )

    if df.empty:
        logger.warning(
            "No reviews were extracted. Downstream transform/load steps will "
            "run on an empty dataset."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Extraction failed.")
        sys.exit(1)