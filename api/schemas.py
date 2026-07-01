"""
api/schemas.py
===============
Pydantic models defining request/response shapes for the FastAPI service.
Kept separate from main.py so route handlers stay focused on logic.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ReviewOut(BaseModel):
    """A single review as returned by GET /reviews."""

    model_config = ConfigDict(from_attributes=True)

    review_id: str
    app_id: str
    user_name: Optional[str] = None
    rating: int
    review_text: Optional[str] = None
    cleaned_text: Optional[str] = None
    sentiment_score: float
    sentiment_label: str
    review_date: dt.datetime
    thumbs_up_count: int
    app_version: Optional[str] = None


class PaginatedReviews(BaseModel):
    """Paginated envelope for GET /reviews."""

    total: int = Field(..., description="Total matching rows across all pages")
    page: int
    page_size: int
    total_pages: int
    items: List[ReviewOut]


class SentimentTrendPoint(BaseModel):
    """One day's sentiment counts, for GET /sentiment-trend."""

    date: dt.date
    positive: int
    neutral: int
    negative: int


class KeywordOut(BaseModel):
    """A single keyword/frequency pair, for GET /top-keywords."""

    keyword: str
    frequency: int


class SummaryOut(BaseModel):
    """Aggregate KPI snapshot, for GET /summary."""

    total_reviews: int
    average_rating: float
    positive_count: int
    neutral_count: int
    negative_count: int
    positive_pct: float
    neutral_pct: float
    negative_pct: float


class EtlRunOut(BaseModel):
    """A single pipeline execution record, for GET /etl-stats."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    run_timestamp: dt.datetime
    app_ids: str
    raw_count: int
    duplicates_removed: int
    nulls_handled: int
    final_row_count: int
    new_reviews_inserted: int
    duplicate_reviews_skipped: int
    status: str
    error_message: Optional[str] = None


class EtlStatsOut(BaseModel):
    """Latest run plus recent run history, for GET /etl-stats."""

    latest_run: Optional[EtlRunOut] = None
    recent_runs: List[EtlRunOut] = Field(default_factory=list)


class HealthOut(BaseModel):
    """Service health, for GET /health."""

    status: str
    database: str

class AnalyzeAppRequest(BaseModel):
    """Request body for POST /apps/analyze."""

    input: str = Field(..., description="A Play Store URL or a raw Android app id")


class AnalyzeAppResponse(BaseModel):
    """Response for POST /apps/analyze."""

    app_id: str
    status: str
    raw_count: int
    final_row_count: int
    new_reviews_inserted: int
    duplicate_reviews_skipped: int
    message: str