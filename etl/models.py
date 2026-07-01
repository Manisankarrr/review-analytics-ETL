"""
etl/models.py
==============
SQLAlchemy ORM models defining the database schema.

Tables
------
reviews   : one row per Google Play review. `review_id` (Play Store's own
            unique id) is the primary key, which is what makes duplicate
            insertion structurally impossible rather than just "checked for".
keywords  : cumulative keyword frequency per (app_id, keyword, sentiment
            label), upserted on every ETL run.
etl_runs  : one row per pipeline execution, recording extraction/transform
            statistics. This is what powers the Data Quality dashboard page
            via the API (the dashboard never touches the database directly).
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    DateTime,
    Float,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class Review(Base):
    """A single Google Play Store review, cleaned and scored."""

    __tablename__ = "reviews"

    review_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    app_id: Mapped[str] = mapped_column(String(255), nullable=False)
    user_name: Mapped[str] = mapped_column(String(255), nullable=True)
    rating: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    review_text: Mapped[str] = mapped_column(Text, nullable=True)
    cleaned_text: Mapped[str] = mapped_column(Text, nullable=True)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False)
    sentiment_label: Mapped[str] = mapped_column(String(16), nullable=False)
    review_date: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    thumbs_up_count: Mapped[int] = mapped_column(Integer, default=0)
    app_version: Mapped[str] = mapped_column(String(64), nullable=True)
    loaded_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_reviews_app_id", "app_id"),
        Index("ix_reviews_rating", "rating"),
        Index("ix_reviews_sentiment_label", "sentiment_label"),
        Index("ix_reviews_review_date", "review_date"),
    )


class Keyword(Base):
    """Cumulative keyword frequency, scoped by app and sentiment label."""

    __tablename__ = "keywords"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    app_id: Mapped[str] = mapped_column(String(255), nullable=False)
    keyword: Mapped[str] = mapped_column(String(128), nullable=False)
    sentiment_label: Mapped[str] = mapped_column(String(16), nullable=False)
    frequency: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint(
            "app_id", "keyword", "sentiment_label", name="uq_keyword_scope"
        ),
        Index("ix_keywords_frequency", "frequency"),
    )


class EtlRun(Base):
    """One row per pipeline execution; backs the Data Quality dashboard."""

    __tablename__ = "etl_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_timestamp: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )
    app_ids: Mapped[str] = mapped_column(String(512), nullable=False)
    raw_count: Mapped[int] = mapped_column(Integer, default=0)
    duplicates_removed: Mapped[int] = mapped_column(Integer, default=0)
    nulls_handled: Mapped[int] = mapped_column(Integer, default=0)
    final_row_count: Mapped[int] = mapped_column(Integer, default=0)
    new_reviews_inserted: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_reviews_skipped: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_etl_runs_run_timestamp", "run_timestamp"),)