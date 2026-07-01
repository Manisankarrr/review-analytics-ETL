"""
etl/load.py
============
Load stage: persists the transformed CSV outputs into Aiven MySQL.

Run as a module from the project root:
    python -m etl.load
"""

from __future__ import annotations

import math
import datetime as dt
import json
import logging
import sys
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Session

from etl.config import Settings, get_settings
from etl.db import get_engine, get_session_factory
from etl.models import Base, EtlRun, Keyword, Review

logger = logging.getLogger(__name__)

INSERT_CHUNK_SIZE = 500


def ensure_tables_exist() -> None:
    """Create any missing tables. Safe to call on every run (idempotent)."""
    Base.metadata.create_all(get_engine())
    logger.info("Verified database schema (tables created if missing).")


def _existing_review_ids(session: Session, review_ids: List[str]) -> set:
    """Return the subset of `review_ids` already present in the reviews table."""
    if not review_ids:
        return set()
    existing = set()
    for i in range(0, len(review_ids), INSERT_CHUNK_SIZE):
        chunk = review_ids[i : i + INSERT_CHUNK_SIZE]
        stmt = select(Review.review_id).where(Review.review_id.in_(chunk))
        existing.update(row[0] for row in session.execute(stmt))
    return existing


def load_reviews(session: Session, reviews_df: pd.DataFrame) -> Dict[str, int]:
    
    if reviews_df.empty:
        return {"new_reviews_inserted": 0, "duplicate_reviews_skipped": 0}

    all_ids = reviews_df["review_id"].astype(str).tolist()
    existing_ids = _existing_review_ids(session, all_ids)

    new_rows_df = reviews_df[~reviews_df["review_id"].astype(str).isin(existing_ids)]
    duplicate_count = len(reviews_df) - len(new_rows_df)

    if new_rows_df.empty:
        logger.info("No new reviews to insert (%d already present).", duplicate_count)
        return {
            "new_reviews_inserted": 0,
            "duplicate_reviews_skipped": duplicate_count,
        }

    records = new_rows_df.to_dict(orient="records")
    for record in records:
        record["rating"] = int(record["rating"])
        record["thumbs_up_count"] = int(record["thumbs_up_count"])
        record["sentiment_score"] = float(record["sentiment_score"])
        review_date = record["review_date"]
        record["review_date"] = (
            review_date.to_pydatetime()
            if isinstance(review_date, pd.Timestamp)
            else review_date
        )
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None

    for i in range(0, len(records), INSERT_CHUNK_SIZE):
        chunk = records[i : i + INSERT_CHUNK_SIZE]
        session.bulk_insert_mappings(Review, chunk)
        logger.info("Inserted review chunk of %d rows.", len(chunk))

    return {
        "new_reviews_inserted": len(new_rows_df),
        "duplicate_reviews_skipped": duplicate_count,
    }


def load_keywords(session: Session, keywords_df: pd.DataFrame) -> int:
    """
    Upsert keyword frequencies: insert new (app_id, keyword, sentiment_label)
    rows, or increment `frequency` on existing ones via
    INSERT ... ON DUPLICATE KEY UPDATE.
    """
    if keywords_df.empty:
        return 0

    records = keywords_df.to_dict(orient="records")
    for record in records:
        record["frequency"] = int(record["frequency"])

    processed = 0
    for i in range(0, len(records), INSERT_CHUNK_SIZE):
        chunk = records[i : i + INSERT_CHUNK_SIZE]
        stmt = mysql_insert(Keyword).values(chunk)
        stmt = stmt.on_duplicate_key_update(
            frequency=Keyword.frequency + stmt.inserted.frequency,
            updated_at=dt.datetime.utcnow(),
        )
        session.execute(stmt)
        processed += len(chunk)
        logger.info("Upserted keyword chunk of %d rows.", len(chunk))

    return processed


def record_etl_run(
    session: Session,
    app_ids: List[str],
    stats: dict,
    load_counts: Dict[str, int],
    status: str,
    error_message: Optional[str] = None,
) -> None:
    """Insert a row into etl_runs summarizing this pipeline execution."""
    run = EtlRun(
        run_timestamp=dt.datetime.utcnow(),
        app_ids=",".join(app_ids),
        raw_count=stats.get("raw_count", 0),
        duplicates_removed=stats.get("duplicates_removed", 0),
        nulls_handled=stats.get("nulls_handled", 0),
        final_row_count=stats.get("final_row_count", 0),
        new_reviews_inserted=load_counts.get("new_reviews_inserted", 0),
        duplicate_reviews_skipped=load_counts.get("duplicate_reviews_skipped", 0),
        status=status,
        error_message=error_message,
    )
    session.add(run)


def load_dataframes(
    app_ids: List[str],
    reviews_df: pd.DataFrame,
    keywords_df: pd.DataFrame,
    stats: dict,
) -> Dict[str, int]:
    """
    Transactionally load already-transformed DataFrames into the database.

    This is the shared transactional core used by both:
      * `run_load()` — the disk-based CLI/GitHub Actions flow
      * the API's on-demand single-app analysis flow, which has its
        DataFrames already in memory and never touches disk.
    """
    ensure_tables_exist()
    SessionFactory = get_session_factory()
    session = SessionFactory()
    load_counts: Dict[str, int] = {}
    try:
        load_counts = load_reviews(session, reviews_df)
        load_keywords(session, keywords_df)
        record_etl_run(session, app_ids, stats, load_counts, status="SUCCESS")
        session.commit()
        logger.info(
            "Load committed successfully. new_reviews_inserted=%d "
            "duplicate_reviews_skipped=%d",
            load_counts.get("new_reviews_inserted", 0),
            load_counts.get("duplicate_reviews_skipped", 0),
        )
        return load_counts
    except Exception as exc:
        logger.exception("Load failed; rolling back transaction.")
        session.rollback()
        try:
            failure_session = SessionFactory()
            record_etl_run(
                failure_session,
                app_ids,
                stats,
                load_counts,
                status="FAILED",
                error_message=str(exc)[:2000],
            )
            failure_session.commit()
            failure_session.close()
        except Exception:
            logger.exception("Additionally failed to record the FAILED etl_run row.")
        raise
    finally:
        session.close()


def run_load(settings: Settings) -> Dict[str, int]:
    """Orchestrate the load stage from disk-based ETL stage outputs."""
    if not settings.processed_reviews_path.exists():
        raise FileNotFoundError(
            f"{settings.processed_reviews_path} not found. "
            "Run `python -m etl.transform` first."
        )

    reviews_df = pd.read_csv(settings.processed_reviews_path)
    keywords_df = (
        pd.read_csv(settings.processed_keywords_path)
        if settings.processed_keywords_path.exists()
        else pd.DataFrame(columns=["app_id", "keyword", "sentiment_label", "frequency"])
    )
    stats = (
        json.loads(settings.etl_stats_path.read_text())
        if settings.etl_stats_path.exists()
        else {}
    )

    app_ids_str = stats.get("app_ids", ",".join(settings.google_play_app_ids))
    app_ids = [a for a in app_ids_str.split(",") if a]

    return load_dataframes(app_ids, reviews_df, keywords_df, stats)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = get_settings()
    run_load(settings)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Load stage failed.")
        sys.exit(1)