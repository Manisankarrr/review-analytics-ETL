"""
etl/transform.py
==================
Transform stage: takes the raw extracted CSV and produces a cleaned,
sentiment-scored dataset plus a cumulative-ready keyword frequency table.

Run as a module from the project root:
    python -m etl.transform
"""

from __future__ import annotations

import datetime as dt
import html
import json
import logging
import re
import sys
from typing import List, Tuple

import emoji
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from etl.config import Settings, get_settings

logger = logging.getLogger(__name__)

_REQUIRED_OUTPUT_COLUMNS = [
    "review_id",
    "app_id",
    "user_name",
    "rating",
    "review_text",
    "cleaned_text",
    "sentiment_score",
    "sentiment_label",
    "review_date",
    "thumbs_up_count",
    "app_version",
]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE_RE = re.compile(r"\s+")

_VADER = SentimentIntensityAnalyzer()

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

TOP_KEYWORDS_PER_GROUP = 30
MIN_REVIEWS_FOR_KEYWORDS = 2


def clean_text(raw_text: str) -> str:
    """
    Clean a single review's text:
      1. Unescape HTML entities
      2. Strip HTML tags
      3. Strip URLs
      4. Strip emojis
      5. Collapse repeated whitespace and trim
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return ""

    text = html.unescape(raw_text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = emoji.replace_emoji(text, replace=" ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def score_sentiment(cleaned_text: str) -> float:
    """Return VADER's compound sentiment score in [-1.0, 1.0]."""
    if not cleaned_text:
        return 0.0
    return _VADER.polarity_scores(cleaned_text)["compound"]


def label_sentiment(score: float) -> str:
    """Bucket a compound VADER score into Positive / Neutral / Negative."""
    if score >= POSITIVE_THRESHOLD:
        return "Positive"
    if score <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def extract_top_keywords(texts: List[str], top_n: int = TOP_KEYWORDS_PER_GROUP) -> List[Tuple[str, int]]:
    """
    Identify the most distinctive keywords/phrases using TF-IDF for term
    selection, then report each term's document frequency (number of
    reviews it appears in) as an interpretable count.
    """
    non_empty = [t for t in texts if t and t.strip()]
    if len(non_empty) < MIN_REVIEWS_FOR_KEYWORDS:
        return []

    vectorizer = TfidfVectorizer(
        max_features=top_n,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
        lowercase=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(non_empty)
    except ValueError:
        logger.warning("TF-IDF vectorizer produced an empty vocabulary; skipping.")
        return []

    terms = vectorizer.get_feature_names_out()
    doc_frequency = (tfidf_matrix > 0).sum(axis=0).A1

    term_freq = sorted(zip(terms, doc_frequency), key=lambda x: x[1], reverse=True)
    return [(term, int(freq)) for term, freq in term_freq if freq > 0]


def _normalize_dates(series: pd.Series) -> pd.Series:
    """
    Parse review dates robustly, coercing unparseable values to NaT.

    Uses format="mixed" so each element is parsed independently. Without
    this, pandas' vectorized format inference can lock onto the format of
    the first non-null value and then incorrectly mark valid dates in a
    different format (e.g. date-only vs date+time) as NaT for the rest of
    the column.
    """
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _normalize_ratings(series: pd.Series) -> pd.Series:
    """Coerce ratings to integers in the valid 1-5 range."""
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.clip(lower=1, upper=5)
    return numeric


def transform(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run the full transformation pipeline on the raw extracted DataFrame.

    Returns
    -------
    (clean_reviews_df, keywords_df, stats)
    """
    stats = {
        "raw_count": int(len(raw_df)),
        "duplicates_removed": 0,
        "nulls_handled": 0,
        "final_row_count": 0,
    }

    if raw_df.empty:
        logger.warning("Transform received an empty DataFrame; nothing to do.")
        empty_reviews = pd.DataFrame(columns=_REQUIRED_OUTPUT_COLUMNS)
        empty_keywords = pd.DataFrame(
            columns=["app_id", "keyword", "sentiment_label", "frequency"]
        )
        return empty_reviews, empty_keywords, stats

    df = raw_df.rename(
        columns={
            "reviewId": "review_id",
            "userName": "user_name",
            "content": "review_text",
            "score": "rating",
            "thumbsUpCount": "thumbs_up_count",
            "reviewCreatedVersion": "app_version_legacy",
            "at": "review_date",
            "appVersion": "app_version",
        }
    ).copy()

    if "app_version_legacy" in df.columns:
        df["app_version"] = df["app_version"].fillna(df["app_version_legacy"])
        df = df.drop(columns=["app_version_legacy"])

    # --- 1. Remove duplicate reviews (same review_id) ---
    before = len(df)
    df = df.drop_duplicates(subset=["review_id"], keep="first")
    stats["duplicates_removed"] = before - len(df)

    # --- 2. Drop rows with no review_id (cannot be a primary key) ---
    df = df[df["review_id"].notna()]

    # --- 3. Normalize dates and ratings ---
    df["review_date"] = _normalize_dates(df["review_date"])
    df["rating"] = _normalize_ratings(df["rating"])

    # --- 4. Handle nulls ---
    null_mask_before = df.isna()
    df["user_name"] = df["user_name"].fillna("Anonymous")
    df["thumbs_up_count"] = pd.to_numeric(
        df["thumbs_up_count"], errors="coerce"
    ).fillna(0).astype(int)
    df["app_version"] = df["app_version"].fillna("Unknown")
    df["review_text"] = df["review_text"].fillna("")

    unusable_mask = (df["review_text"].str.strip() == "") & (df["rating"].isna())
    nulls_handled = int(null_mask_before.sum().sum())
    stats["nulls_handled"] = nulls_handled
    df = df[~unusable_mask]

    if df["rating"].isna().any():
        median_rating = df["rating"].median()
        df["rating"] = df["rating"].fillna(median_rating)
    df["rating"] = df["rating"].astype(int)

    df = df[df["review_date"].notna()]

    # --- 5. Clean text ---
    df["cleaned_text"] = df["review_text"].apply(clean_text)

    # --- 6. Sentiment scoring ---
    df["sentiment_score"] = df["cleaned_text"].apply(score_sentiment)
    df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)

    df = df.reset_index(drop=True)
    stats["final_row_count"] = int(len(df))

    # --- 7. Schema validation before returning ---
    missing_cols = [c for c in _REQUIRED_OUTPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Transformed DataFrame is missing columns: {missing_cols}")
    if df[_REQUIRED_OUTPUT_COLUMNS].isna().all(axis=None) and not df.empty:
        raise ValueError("Transformed DataFrame is unexpectedly entirely null.")

    clean_reviews_df = df[_REQUIRED_OUTPUT_COLUMNS].copy()

    # --- 8. Keyword extraction (overall + per sentiment label, per app) ---
    keyword_rows = []
    for app_id, app_group in clean_reviews_df.groupby("app_id"):
        for sentiment_label, group in app_group.groupby("sentiment_label"):
            top_terms = extract_top_keywords(group["cleaned_text"].tolist())
            for term, freq in top_terms:
                keyword_rows.append(
                    {
                        "app_id": app_id,
                        "keyword": term,
                        "sentiment_label": sentiment_label,
                        "frequency": freq,
                    }
                )

    keywords_df = pd.DataFrame(
        keyword_rows, columns=["app_id", "keyword", "sentiment_label", "frequency"]
    )

    return clean_reviews_df, keywords_df, stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings: Settings = get_settings()

    if not settings.raw_data_path.exists():
        logger.error(
            "Raw data file not found at %s. Run `python -m etl.extract` first.",
            settings.raw_data_path,
        )
        sys.exit(1)

    logger.info("Loading raw data from %s", settings.raw_data_path)
    raw_df = pd.read_csv(settings.raw_data_path)

    clean_reviews_df, keywords_df, stats = transform(raw_df)

    settings.processed_reviews_path.parent.mkdir(parents=True, exist_ok=True)
    settings.processed_keywords_path.parent.mkdir(parents=True, exist_ok=True)
    settings.etl_stats_path.parent.mkdir(parents=True, exist_ok=True)

    clean_reviews_df.to_csv(settings.processed_reviews_path, index=False)
    keywords_df.to_csv(settings.processed_keywords_path, index=False)

    stats_payload = {
        **stats,
        "app_ids": ",".join(settings.google_play_app_ids),
        "run_timestamp": dt.datetime.utcnow().isoformat(),
        "keyword_rows_generated": int(len(keywords_df)),
    }
    with open(settings.etl_stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats_payload, fh, indent=2)

    logger.info(
        "Transform complete. raw=%d duplicates_removed=%d nulls_handled=%d "
        "final=%d keywords=%d",
        stats["raw_count"],
        stats["duplicates_removed"],
        stats["nulls_handled"],
        stats["final_row_count"],
        len(keywords_df),
    )
    logger.info(
        "Wrote %s, %s, %s",
        settings.processed_reviews_path,
        settings.processed_keywords_path,
        settings.etl_stats_path,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Transform failed.")
        sys.exit(1)