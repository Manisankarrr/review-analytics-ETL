"""
dashboard/app.py
=================
Enterprise-style analytics dashboard for the Review Intelligence pipeline.

This app NEVER touches the database directly — every figure on screen is
derived from calls to the FastAPI service (api/main.py). Run with:

    streamlit run dashboard/app.py

Configuration (read from environment / .env):
    API_BASE_URL   Base URL of the FastAPI service (default http://localhost:8000)
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration & constants
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 10

# Semantic palette — used in Python logic only (never injected raw into CSS).
# For UI rendering, all colors are resolved via CSS custom properties so they
# automatically adapt to Streamlit's light / dark theme toggle.
COLOR_POSITIVE = "#22c55e"   # green-500
COLOR_NEUTRAL  = "#f59e0b"   # amber-500
COLOR_NEGATIVE = "#ef4444"   # red-500
COLOR_ACCENT   = "#6366f1"   # indigo-500  (primary brand)
COLOR_ACCENT_LIGHT = "#818cf8"  # indigo-400

SENTIMENT_COLORS = {
    "Positive": COLOR_POSITIVE,
    "Neutral":  COLOR_NEUTRAL,
    "Negative": COLOR_NEGATIVE,
}

PAGES = [
    "Executive Overview",
    "Sentiment Analytics",
    "Keyword Intelligence",
    "Review Explorer",
    "Data Quality & ETL Monitoring",
]

PLOTLY_CONFIG = {"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["select2d", "lasso2d"]}


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def inject_custom_css() -> None:
    """Inject CSS design tokens that match config.toml's dark theme.
    Streamlit owns background/sidebar colours via secondaryBackgroundColor;
    we only override layout, typography, and component-level polish."""
    st.markdown(
        """
        <style>
        /* ── Google Font ──────────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* ── Design tokens — dark theme (mirrors config.toml) ───────── */
        :root {
            --color-accent:         #6366f1;  /* primaryColor in config.toml  */
            --color-accent-hover:   #4f46e5;
            --color-positive:       #22c55e;
            --color-neutral:        #f59e0b;
            --color-negative:       #ef4444;
            /* Surfaces — matches config.toml backgroundColor / secondaryBackgroundColor */
            --color-surface:        #1e293b;  /* slate-800 — cards, metrics   */
            --color-surface-raised: #0f172a;  /* slate-900 — page background  */
            --color-border:         #334155;  /* slate-700                    */
            --color-text-primary:   #e2e8f0;  /* textColor in config.toml     */
            --color-text-secondary: #94a3b8;  /* slate-400                    */
            --color-shadow:         rgba(0, 0, 0, 0.35);
        }

        /* ── Global typography ────────────────────────────────────────── */
        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
        }

        /* ── KPI metric cards ─────────────────────────────────────────── */
        div[data-testid="stMetric"] {
            background: var(--color-surface);
            border: 1px solid var(--color-border);
            border-radius: 14px;
            padding: 20px 20px 16px 20px;
            box-shadow: 0 2px 8px var(--color-shadow);
            transition: box-shadow 0.2s ease, transform 0.15s ease;
            position: relative;
            overflow: hidden;
        }
        div[data-testid="stMetric"]:hover {
            box-shadow: 0 6px 20px var(--color-shadow);
            transform: translateY(-2px);
        }
        div[data-testid="stMetric"]::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--color-accent), var(--color-accent-hover));
            border-radius: 14px 14px 0 0;
        }
        div[data-testid="stMetricLabel"] > div {
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--color-text-secondary) !important;
        }
        div[data-testid="stMetricValue"] > div {
            font-size: 2rem;
            font-weight: 700;
            color: var(--color-text-primary) !important;
            letter-spacing: -0.02em;
        }
        div[data-testid="stMetricDelta"] {
            font-weight: 600;
            font-size: 0.82rem;
        }

        /* ── Sidebar ──────────────────────────────────────────────────── */
        /* Background is controlled by secondaryBackgroundColor in config.toml. */
        /* Only override layout/typography, not the background colour.          */
        section[data-testid="stSidebar"] {
            border-right: 1px solid var(--color-border);
        }
        section[data-testid="stSidebar"] .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 4px 0 16px;
        }
        section[data-testid="stSidebar"] .sidebar-brand h2 {
            margin: 0;
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--color-text-primary);
        }

        /* ── Page header ──────────────────────────────────────────────── */
        .dashboard-header {
            padding: 4px 0 20px 0;
            border-bottom: 1px solid var(--color-border);
            margin-bottom: 24px;
        }
        .dashboard-header h1 {
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: var(--color-text-primary);
            margin-bottom: 4px;
        }
        .dashboard-subtitle {
            color: var(--color-text-secondary);
            font-size: 0.93rem;
            margin: 0;
        }

        /* ── Section divider ──────────────────────────────────────────── */
        .section-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: var(--color-text-secondary);
            margin: 28px 0 12px;
        }

        /* ── Status badges ────────────────────────────────────────────── */
        .status-badge-success {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 14px;
            border-radius: 20px;
            background: rgba(34, 197, 94, 0.12);
            color: #16a34a;
            font-weight: 600;
            font-size: 0.83rem;
            border: 1px solid rgba(34, 197, 94, 0.25);
        }
        .status-badge-failed {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 14px;
            border-radius: 20px;
            background: rgba(239, 68, 68, 0.12);
            color: #dc2626;
            font-weight: 600;
            font-size: 0.83rem;
            border: 1px solid rgba(239, 68, 68, 0.25);
        }

        /* ── Pagination info ──────────────────────────────────────────── */
        .page-info {
            text-align: center;
            color: var(--color-text-secondary);
            font-size: 0.88rem;
            font-weight: 500;
            padding: 8px 0;
        }

        /* ── Dataframe tweaks ─────────────────────────────────────────── */
        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--color-border);
        }

        /* ── Plotly chart containers ──────────────────────────────────── */
        div[data-testid="stPlotlyChart"] {
            border-radius: 12px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Theme is set authoritatively in .streamlit/config.toml.
# Change base="dark" → base="light" there, and flip this constant to match.
PLOTLY_TEMPLATE = "plotly_dark"


def get_plotly_template() -> str:
    """Return the Plotly template that matches .streamlit/config.toml [theme].base."""
    return PLOTLY_TEMPLATE


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
def safe_api_get(
    endpoint: str, params: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Call the FastAPI service and return (data, error_message). Exactly one
    of the two will be non-None. Never raises.
    """
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.get(url, params=params or {}, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, (
            f"Could not connect to the API at {API_BASE_URL}. "
            "Confirm the FastAPI service is running and API_BASE_URL is correct."
        )
    except requests.exceptions.Timeout:
        return None, "The API request timed out. The service may be overloaded."
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        return None, f"API returned an error ({exc.response.status_code}): {detail or exc}"
    except requests.exceptions.RequestException as exc:
        return None, f"Unexpected network error contacting the API: {exc}"
    except ValueError:
        return None, "API returned a response that could not be parsed as JSON."


def safe_api_post(
    endpoint: str, json_body: Optional[Dict[str, Any]] = None, timeout: int = 90
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Call a mutating FastAPI endpoint and return (data, error_message). Never
    cached, since this triggers a real ETL run rather than reading data.
    """
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, json=json_body or {}, timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, (
            f"Could not connect to the API at {API_BASE_URL}. "
            "Confirm the FastAPI service is running and API_BASE_URL is correct."
        )
    except requests.exceptions.Timeout:
        return None, "The request timed out while fetching and analyzing reviews."
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        return None, detail or f"API returned an error ({exc.response.status_code})."
    except requests.exceptions.RequestException as exc:
        return None, f"Unexpected network error contacting the API: {exc}"
    except ValueError:
        return None, "API returned a response that could not be parsed as JSON."


def trigger_app_analysis(user_input: str) -> Tuple[Optional[dict], Optional[str]]:
    """Trigger an on-demand extract → transform → load run for one app."""
    return safe_api_post("/apps/analyze", {"input": user_input.strip()})


@st.cache_data(ttl=120, show_spinner=False)
def fetch_summary(app_id: Optional[str]) -> Tuple[Optional[dict], Optional[str]]:
    params = {"app_id": app_id} if app_id else {}
    return safe_api_get("/summary", params)


@st.cache_data(ttl=120, show_spinner=False)
def fetch_sentiment_trend(app_id: Optional[str]) -> Tuple[Optional[list], Optional[str]]:
    params = {"app_id": app_id} if app_id else {}
    return safe_api_get("/sentiment-trend", params)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_top_keywords(
    sentiment_label: Optional[str], app_id: Optional[str], limit: int
) -> Tuple[Optional[list], Optional[str]]:
    params: Dict[str, Any] = {"limit": limit}
    if sentiment_label:
        params["sentiment_label"] = sentiment_label
    if app_id:
        params["app_id"] = app_id
    return safe_api_get("/top-keywords", params)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_total_keywords_tracked(app_id: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    """Union distinct keywords across the three sentiment buckets (capped at 100 each)."""
    distinct_keywords: set = set()
    for label in ("Positive", "Neutral", "Negative"):
        data, error = fetch_top_keywords(label, app_id, 100)
        if error:
            return None, error
        if data:
            distinct_keywords.update(item["keyword"] for item in data)
    return len(distinct_keywords), None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_reviews(
    page: int,
    page_size: int,
    sentiment_label: Optional[str],
    rating: Optional[int],
    min_rating: Optional[int],
    max_rating: Optional[int],
    app_id: Optional[str],
    search: Optional[str],
    sort_by: str,
    order: str,
) -> Tuple[Optional[dict], Optional[str]]:
    params: Dict[str, Any] = {
        "page": page,
        "page_size": page_size,
        "sort_by": sort_by,
        "order": order,
    }
    if sentiment_label:
        params["sentiment_label"] = sentiment_label
    if rating is not None:
        params["rating"] = rating
    if min_rating is not None:
        params["min_rating"] = min_rating
    if max_rating is not None:
        params["max_rating"] = max_rating
    if app_id:
        params["app_id"] = app_id
    if search:
        params["search"] = search
    return safe_api_get("/reviews", params)


@st.cache_data(ttl=180, show_spinner=False)
def fetch_rating_distribution(app_id: Optional[str]) -> Tuple[Optional[Dict[int, int]], Optional[str]]:
    """Derive a rating histogram by querying /reviews once per rating value."""
    distribution: Dict[int, int] = {}
    for rating in range(1, 6):
        data, error = fetch_reviews(
            page=1,
            page_size=1,
            sentiment_label=None,
            rating=rating,
            min_rating=None,
            max_rating=None,
            app_id=app_id,
            search=None,
            sort_by="review_date",
            order="desc",
        )
        if error:
            return None, error
        distribution[rating] = data["total"] if data else 0
    return distribution, None


@st.cache_data(ttl=120, show_spinner=False)
def fetch_etl_stats(history: int) -> Tuple[Optional[dict], Optional[str]]:
    return safe_api_get("/etl-stats", {"history": history})


def fetch_health() -> Tuple[Optional[dict], Optional[str]]:
    return safe_api_get("/health")


# ---------------------------------------------------------------------------
# Data transformation helpers
# ---------------------------------------------------------------------------
def trend_to_dataframe(trend_points: Optional[List[dict]]) -> pd.DataFrame:
    columns = ["date", "positive", "neutral", "negative", "total",
               "positive_pct", "neutral_pct", "negative_pct"]
    if not trend_points:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(trend_points)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["total"] = df["positive"] + df["neutral"] + df["negative"]

    for col in ("positive", "neutral", "negative"):
        df[f"{col}_pct"] = df.apply(
            lambda row, c=col: round(100 * row[c] / row["total"], 1) if row["total"] > 0 else 0.0,
            axis=1,
        )
    return df


def compute_period_deltas(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Compare the second half of the trend window against the first half."""
    if df.empty or len(df) < 2:
        return {"total_delta_pct": None, "positive_pct_delta": None, "negative_pct_delta": None}

    midpoint = len(df) // 2
    older, recent = df.iloc[:midpoint], df.iloc[midpoint:]

    older_total, recent_total = older["total"].sum(), recent["total"].sum()
    total_delta_pct = (
        round(100 * (recent_total - older_total) / older_total, 1) if older_total > 0 else None
    )

    def _pct(group: pd.DataFrame, col: str) -> Optional[float]:
        total = group["total"].sum()
        return round(100 * group[col].sum() / total, 1) if total > 0 else None

    older_pos, recent_pos = _pct(older, "positive"), _pct(recent, "positive")
    older_neg, recent_neg = _pct(older, "negative"), _pct(recent, "negative")

    positive_pct_delta = (
        round(recent_pos - older_pos, 1) if older_pos is not None and recent_pos is not None else None
    )
    negative_pct_delta = (
        round(recent_neg - older_neg, 1) if older_neg is not None and recent_neg is not None else None
    )

    return {
        "total_delta_pct": total_delta_pct,
        "positive_pct_delta": positive_pct_delta,
        "negative_pct_delta": negative_pct_delta,
    }


def format_relative_timestamp(iso_timestamp: str) -> str:
    try:
        ts = datetime.fromisoformat(iso_timestamp)
    except (TypeError, ValueError):
        return iso_timestamp or "Unknown"
    return ts.strftime("%b %d, %Y at %H:%M UTC")


# ---------------------------------------------------------------------------
# Chart builders (Plotly only)
# ---------------------------------------------------------------------------
def apply_standard_layout(
    fig: go.Figure,
    title: str,
    y_title: str = "",
    height: int = 380,
) -> go.Figure:
    """Apply a consistent, theme-aware layout to any Plotly figure.

    autosize=True is set so Plotly reflows the chart to fill whatever
    column width is available after the sidebar expands or collapses.
    The `height` parameter still sets a *minimum* visual height via the
    Plotly layout; actual width is always driven by the container.
    """
    template = get_plotly_template()
    is_dark = template == "plotly_dark"
    paper_bg  = "rgba(0,0,0,0)"           # transparent — inherits Streamlit background
    plot_bg   = "rgba(0,0,0,0)"
    grid_color = "rgba(148,163,184,0.15)" if is_dark else "rgba(100,116,139,0.12)"
    text_color = "#e2e8f0" if is_dark else "#334155"
    title_color = "#f1f5f9" if is_dark else "#0f172a"

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=15, color=title_color, family="Inter, Segoe UI, sans-serif"),
            x=0,
            pad=dict(l=4),
        ),
        template=template,
        # autosize lets Plotly fill the available column width rather than
        # rendering at a hard-coded pixel width. This prevents chart
        # clipping / overlap when the Streamlit sidebar opens or closes.
        autosize=True,
        height=height,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        # Keep margins lean so charts don't lose rendering area inside
        # narrow columns (e.g. [1, 2] splits with the sidebar open).
        margin=dict(l=40, r=16, t=52, b=40),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=text_color),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        yaxis=dict(
            title=y_title,
            gridcolor=grid_color,
            zerolinecolor=grid_color,
        ),
        xaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=grid_color,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1e293b" if is_dark else "#ffffff",
            bordercolor="#334155" if is_dark else "#e2e8f0",
            font_color="#f1f5f9" if is_dark else "#0f172a",
            font_size=12,
        ),
    )
    return fig


def build_empty_state_figure(message: str, height: int = 380) -> go.Figure:
    template = get_plotly_template()
    is_dark = template == "plotly_dark"
    text_color = "#64748b" if not is_dark else "#94a3b8"
    fig = go.Figure()
    fig.add_annotation(
        text=f"<i>{message}</i>",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color=text_color, family="Inter, Segoe UI, sans-serif"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        template=template,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def build_volume_area_chart(df: pd.DataFrame) -> go.Figure:
    """Filled area chart for daily review volume with a smooth gradient."""
    if df.empty:
        return build_empty_state_figure("No review volume data available yet.")
    fig = go.Figure(
        go.Scatter(
            x=df["date"],
            y=df["total"],
            mode="lines",
            fill="tozeroy",
            line=dict(color=COLOR_ACCENT, width=2.5, shape="spline", smoothing=0.8),
            fillcolor="rgba(99, 102, 241, 0.12)",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>%{y:,} reviews<extra></extra>",
            name="Reviews",
        )
    )
    return apply_standard_layout(fig, "Review Volume Trend", "Reviews per day")


def build_sentiment_trend_lines(df: pd.DataFrame, use_pct: bool = False) -> go.Figure:
    if df.empty:
        return build_empty_state_figure("No sentiment trend data available yet.")
    fig = go.Figure()
    suffix = "_pct" if use_pct else ""
    fmt = ".1f" if use_pct else ","
    unit = "% of reviews" if use_pct else "reviews"
    for label in ("positive", "neutral", "negative"):
        col = label.capitalize()
        color = SENTIMENT_COLORS[col]
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[f"{label}{suffix}"],
                mode="lines+markers",
                name=col,
                line=dict(color=color, width=2.5, shape="spline", smoothing=0.7),
                marker=dict(size=5, symbol="circle", line=dict(width=1.5, color=color)),
                hovertemplate=(
                    f"<b>%{{x|%b %d}}</b><br>{col}: %{{y:{fmt}}} {unit}<extra></extra>"
                ),
            )
        )
    title = "Sentiment Share (%)" if use_pct else "Sentiment Trend Over Time"
    y_title = "% of daily reviews" if use_pct else "Reviews"
    fig = apply_standard_layout(fig, title, y_title)
    if use_pct:
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return fig


def build_sentiment_donut(positive: int, neutral: int, negative: int) -> go.Figure:
    if positive + neutral + negative == 0:
        return build_empty_state_figure("No sentiment data available yet.", height=360)
    total = positive + neutral + negative
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Positive", "Neutral", "Negative"],
                values=[positive, neutral, negative],
                hole=0.62,
                marker=dict(
                    colors=[COLOR_POSITIVE, COLOR_NEUTRAL, COLOR_NEGATIVE],
                    line=dict(color="rgba(0,0,0,0)", width=2),
                ),
                textinfo="percent",
                textfont=dict(size=12, family="Inter, Segoe UI, sans-serif"),
                hovertemplate="<b>%{label}</b><br>%{value:,} reviews (%{percent})<extra></extra>",
                pull=[0.03, 0, 0],
            )
        ]
    )
    fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='font-size:10px'>reviews</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, family="Inter, Segoe UI, sans-serif"),
        xref="paper", yref="paper",
    )
    return apply_standard_layout(fig, "Sentiment Distribution", height=360)


def build_rating_distribution_chart(distribution: Dict[int, int]) -> go.Figure:
    if not distribution or sum(distribution.values()) == 0:
        return build_empty_state_figure("No rating data available yet.")
    ratings = sorted(distribution.keys())
    star_colors = ["#ef4444", "#f97316", "#f59e0b", "#84cc16", "#22c55e"]
    fig = go.Figure(
        go.Bar(
            x=[f"{r} ★" for r in ratings],
            y=[distribution[r] for r in ratings],
            marker=dict(
                color=star_colors,
                line=dict(width=0),
                cornerradius=6,
            ),
            hovertemplate="<b>%{x}</b><br>%{y:,} reviews<extra></extra>",
            text=[f"{distribution[r]:,}" for r in ratings],
            textposition="outside",
            textfont=dict(size=11),
        )
    )
    return apply_standard_layout(fig, "Rating Distribution", "Reviews")


def build_sentiment_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return build_empty_state_figure("No data available for a sentiment heatmap yet.", height=320)
    dates = df["date"].dt.strftime("%b %d").tolist()
    z = [df["positive"].tolist(), df["neutral"].tolist(), df["negative"].tolist()]
    is_dark = get_plotly_template() == "plotly_dark"
    scale = [
        [0.0, "rgba(99,102,241,0.04)"],
        [0.5, "rgba(99,102,241,0.45)"],
        [1.0, "#6366f1"],
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=dates,
            y=["Positive", "Neutral", "Negative"],
            colorscale=scale,
            hovertemplate="<b>%{y}</b> on %{x}<br>%{z:,} reviews<extra></extra>",
            colorbar=dict(
                title=dict(text="Reviews", side="right"),
                thickness=12,
                len=0.8,
            ),
            xgap=2,
            ygap=2,
        )
    )
    return apply_standard_layout(fig, "Sentiment Intensity Heatmap", height=320)


def build_keyword_bar_chart(keywords: Optional[List[dict]], color: str, title: str) -> go.Figure:
    if not keywords:
        return build_empty_state_figure(f"No keywords available for {title.lower()}.", height=420)
    df = pd.DataFrame(keywords).sort_values("frequency")
    max_freq = df["frequency"].max() or 1
    bar_colors = [
        f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},{0.4 + 0.6 * v / max_freq:.2f})"
        for c, v in zip([color] * len(df), df["frequency"])
    ]
    fig = go.Figure(
        go.Bar(
            x=df["frequency"],
            y=df["keyword"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0), cornerradius=4),
            hovertemplate="<b>%{y}</b><br>%{x:,} occurrences<extra></extra>",
            text=df["frequency"],
            textposition="outside",
            textfont=dict(size=10),
        )
    )
    chart_height = max(360, 30 * len(df))
    return apply_standard_layout(fig, title, "Frequency", height=chart_height)


def build_etl_run_history_chart(runs: List[dict]) -> go.Figure:
    if not runs:
        return build_empty_state_figure("No ETL run history available yet.")
    df = pd.DataFrame(runs)
    df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])
    df = df.sort_values("run_timestamp")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["run_timestamp"],
            y=df["new_reviews_inserted"],
            name="New Reviews Inserted",
            marker=dict(color=COLOR_POSITIVE, line=dict(width=0), cornerradius=4),
            hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>%{y:,} new reviews<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["run_timestamp"],
            y=df["duplicate_reviews_skipped"],
            name="Duplicates Skipped",
            marker=dict(color=COLOR_NEUTRAL, opacity=0.7, line=dict(width=0), cornerradius=4),
            hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>%{y:,} duplicates skipped<extra></extra>",
        )
    )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    return apply_standard_layout(fig, "ETL Run History", "Reviews")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> Tuple[str, Optional[str]]:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <span style="font-size:1.5rem"></span>
            <h2>Review Intelligence</h2>
        </div>
        <p style="color:var(--color-text-secondary);font-size:0.82rem;margin:-8px 0 0;">Google Play Analytics</p>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    st.sidebar.markdown('<p class="section-label">Navigation</p>', unsafe_allow_html=True)
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

    st.sidebar.divider()
    st.sidebar.markdown('<p class="section-label">Analyze a New App</p>', unsafe_allow_html=True)
    analyze_input = st.sidebar.text_input(
        "App ID or Play Store URL",
        value="",
        placeholder="com.netflix.mediaclient or a Play Store link",
        help="Paste a raw Android app id, or a full Play Store URL — the app id is extracted automatically.",
        key="analyze_input",
    )
    if st.sidebar.button("Analyze Reviews", width="stretch", key="analyze_button"):
        if not analyze_input.strip():
            st.sidebar.error("Enter an app id or Play Store URL first.")
        else:
            with st.spinner("Fetching and analyzing reviews..."):
                result, error = trigger_app_analysis(analyze_input)
            if error:
                st.sidebar.error(error)
            else:
                st.sidebar.success(result.get("message", "Analysis complete."))
                st.session_state["app_id_filter"] = result["app_id"]
                st.cache_data.clear()
                st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown('<p class="section-label">Filters</p>', unsafe_allow_html=True)
    app_id_input = st.sidebar.text_input(
        "App ID",
        value=st.session_state.get("app_id_filter", ""),
        placeholder="e.g. com.spotify.music",
        help="Leave blank to include all tracked apps.",
        key="app_id_filter",
    )
    app_id = app_id_input.strip() or None

    st.sidebar.divider()
    if st.sidebar.button("🔄 Refresh Data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    health_data, health_error = fetch_health()
    if health_error:
        st.sidebar.error("API unreachable")
    elif health_data:
        db_status = health_data.get("database", "unknown")
        if db_status == "connected":
            st.sidebar.success("API & database online")
        else:
            st.sidebar.warning("API online, database unreachable")

    st.sidebar.caption(f"API: {API_BASE_URL}")
    return page, app_id

# ---------------------------------------------------------------------------
# Page: Executive Overview
# ---------------------------------------------------------------------------
def render_executive_overview(app_id: Optional[str]) -> None:
    st.markdown(
        '<div class="dashboard-header"><h1>Executive Overview</h1>'
        '<p class="dashboard-subtitle">Headline performance across all tracked reviews.</p></div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading executive summary..."):
        summary, summary_error = fetch_summary(app_id)
        trend_raw, trend_error = fetch_sentiment_trend(app_id)
        keyword_count, keyword_error = fetch_total_keywords_tracked(app_id)

    if summary_error:
        st.error(summary_error)
        return

    trend_df = trend_to_dataframe(trend_raw) if not trend_error else pd.DataFrame()
    if trend_error:
        st.warning(f"Trend data unavailable: {trend_error}")
    deltas = compute_period_deltas(trend_df)

    if not summary or summary.get("total_reviews", 0) == 0:
        st.info("No reviews loaded yet. Run the ETL pipeline to populate the dashboard.")
        return

    # Row 1 — volume & rating KPIs
    st.markdown('<p class="section-label">Volume &amp; Quality</p>', unsafe_allow_html=True)
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(
        "Total Reviews",
        f"{summary['total_reviews']:,}",
        delta=f"{deltas['total_delta_pct']:+.1f}% vs prior period" if deltas["total_delta_pct"] is not None else None,
    )
    kpi2.metric("Avg. Star Rating", f"{summary['average_rating']:.2f} / 5.00")
    kpi3.metric(
        "Keywords Tracked",
        f"{keyword_count:,}" if keyword_count is not None else "—",
    )
    if keyword_error:
        st.caption(f"Keyword count unavailable: {keyword_error}")

    # Row 2 — sentiment KPIs
    st.markdown('<p class="section-label">Sentiment Breakdown</p>', unsafe_allow_html=True)
    skpi1, skpi2, skpi3 = st.columns(3)
    skpi1.metric(
        "Positive Reviews",
        f"{summary['positive_pct']:.1f}%",
        delta=f"{deltas['positive_pct_delta']:+.1f} pts" if deltas["positive_pct_delta"] is not None else None,
    )
    skpi2.metric("Neutral Reviews", f"{summary['neutral_pct']:.1f}%")
    skpi3.metric(
        "Negative Reviews",
        f"{summary['negative_pct']:.1f}%",
        delta=f"{deltas['negative_pct_delta']:+.1f} pts" if deltas["negative_pct_delta"] is not None else None,
        delta_color="inverse",
    )

    st.markdown('<p class="section-label">Trend Overview</p>', unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(
            build_volume_area_chart(trend_df),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with chart_col2:
        st.plotly_chart(
            build_sentiment_trend_lines(trend_df),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    st.markdown('<p class="section-label">Distribution</p>', unsafe_allow_html=True)
    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        with st.spinner("Loading rating distribution..."):
            distribution, dist_error = fetch_rating_distribution(app_id)
        if dist_error:
            st.warning(f"Rating distribution unavailable: {dist_error}")
        else:
            st.plotly_chart(
                build_rating_distribution_chart(distribution),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
    with chart_col4:
        st.plotly_chart(
            build_sentiment_donut(
                summary["positive_count"], summary["neutral_count"], summary["negative_count"]
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )


# ---------------------------------------------------------------------------
# Page: Sentiment Analytics
# ---------------------------------------------------------------------------
def render_sentiment_analytics(app_id: Optional[str]) -> None:
    st.markdown(
        '<div class="dashboard-header"><h1>Sentiment Analytics</h1>'
        '<p class="dashboard-subtitle">Deep dive into how sentiment is distributed and evolving.</p></div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading sentiment analytics..."):
        summary, summary_error = fetch_summary(app_id)
        trend_raw, trend_error = fetch_sentiment_trend(app_id)

    if summary_error:
        st.error(summary_error)
        return
    if trend_error:
        st.error(trend_error)
        return

    trend_df = trend_to_dataframe(trend_raw)

    if not summary or summary.get("total_reviews", 0) == 0:
        st.info("No reviews loaded yet. Run the ETL pipeline to populate the dashboard.")
        return

    # Stack the donut above the trend line on the Sentiment Analytics page.
    # The previous [1, 2] column split caused the donut to compress badly
    # when the sidebar was open on medium-width screens. Full-width rows
    # let both charts use all available space and resize freely.
    st.plotly_chart(
        build_sentiment_donut(
            summary["positive_count"], summary["neutral_count"], summary["negative_count"]
        ),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    st.plotly_chart(
        build_sentiment_trend_lines(trend_df),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.plotly_chart(
        build_sentiment_heatmap(trend_df),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    st.plotly_chart(
        build_sentiment_trend_lines(trend_df, use_pct=True),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


# ---------------------------------------------------------------------------
# Page: Keyword Intelligence
# ---------------------------------------------------------------------------
def render_keyword_intelligence(app_id: Optional[str]) -> None:
    st.markdown(
        '<div class="dashboard-header"><h1>Keyword Intelligence</h1>'
        '<p class="dashboard-subtitle">What reviewers are actually talking about.</p></div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading keyword data..."):
        positive_kw, pos_error = fetch_top_keywords("Positive", app_id, 15)
        negative_kw, neg_error = fetch_top_keywords("Negative", app_id, 15)
        neutral_kw, neu_error = fetch_top_keywords("Neutral", app_id, 15)

    col1, col2 = st.columns(2)
    with col1:
        if pos_error:
            st.warning(pos_error)
        st.plotly_chart(
            build_keyword_bar_chart(positive_kw, COLOR_POSITIVE, "Top Positive Keywords"),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with col2:
        if neg_error:
            st.warning(neg_error)
        st.plotly_chart(
            build_keyword_bar_chart(negative_kw, COLOR_NEGATIVE, "Top Negative Keywords"),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    if neu_error:
        st.warning(neu_error)
    st.plotly_chart(
        build_keyword_bar_chart(neutral_kw, COLOR_NEUTRAL, "Top Neutral Keywords"),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.markdown("#### Interactive Keyword Exploration")
    explore_col1, explore_col2 = st.columns([1, 1])
    with explore_col1:
        explore_sentiment = st.selectbox(
            "Sentiment category", ["Positive", "Neutral", "Negative"], key="kw_explore_sentiment"
        )
    with explore_col2:
        explore_limit = st.slider("Number of keywords", min_value=5, max_value=50, value=20, key="kw_explore_limit")

    with st.spinner("Loading keyword exploration data..."):
        explore_data, explore_error = fetch_top_keywords(explore_sentiment, app_id, explore_limit)

    if explore_error:
        st.error(explore_error)
        return

    st.plotly_chart(
        build_keyword_bar_chart(
            explore_data, SENTIMENT_COLORS[explore_sentiment], f"Top {explore_sentiment} Keywords (Explorer)"
        ),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    if explore_data:
        st.dataframe(
            pd.DataFrame(explore_data).rename(columns={"keyword": "Keyword", "frequency": "Frequency"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No keywords found for this selection yet.")


# ---------------------------------------------------------------------------
# Page: Review Explorer
# ---------------------------------------------------------------------------
EXPLORER_PAGE_SIZE = 10
EXPLORER_FETCH_SIZE = 200  # API maximum page_size; date filtering is applied client-side on this batch.


def render_review_explorer(app_id: Optional[str]) -> None:
    st.markdown(
        '<div class="dashboard-header"><h1>Review Explorer</h1>'
        '<p class="dashboard-subtitle">Search and filter individual reviews.</p></div>',
        unsafe_allow_html=True,
    )

    # Row 1: two columns so filters don't compress when the sidebar is open.
    # Four equal columns on a wide screen become two overlapping panes on
    # narrow viewports — splitting into two rows of two keeps each control
    # readable at all sidebar states.
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        search_text = st.text_input("Search reviews / keywords", value="", key="explorer_search")
    with filter_col2:
        sentiment_filter = st.selectbox(
            "Sentiment", ["All", "Positive", "Neutral", "Negative"], key="explorer_sentiment"
        )

    # Row 2: remaining filter controls
    filter_col3, filter_col4 = st.columns(2)
    with filter_col3:
        rating_filter = st.selectbox("Rating", ["All", 1, 2, 3, 4, 5], key="explorer_rating")
    with filter_col4:
        sort_by = st.selectbox(
            "Sort by", ["review_date", "rating", "sentiment_score", "thumbs_up_count"], key="explorer_sort"
        )

    # Row 3: date range + sort order
    date_col1, date_col2, order_col = st.columns(3)
    with date_col1:
        date_from = st.date_input("From date", value=None, key="explorer_date_from")
    with date_col2:
        date_to = st.date_input("To date", value=None, key="explorer_date_to")
    with order_col:
        order = st.selectbox("Order", ["desc", "asc"], key="explorer_order")

    filter_signature = (search_text, sentiment_filter, rating_filter, sort_by, order, app_id, str(date_from), str(date_to))
    if st.session_state.get("explorer_filter_signature") != filter_signature:
        st.session_state["explorer_filter_signature"] = filter_signature
        st.session_state["explorer_page"] = 1

    sentiment_param = None if sentiment_filter == "All" else sentiment_filter
    rating_param = None if rating_filter == "All" else int(rating_filter)

    with st.spinner("Loading reviews..."):
        data, error = fetch_reviews(
            page=1,
            page_size=EXPLORER_FETCH_SIZE,
            sentiment_label=sentiment_param,
            rating=rating_param,
            min_rating=None,
            max_rating=None,
            app_id=app_id,
            search=search_text or None,
            sort_by=sort_by,
            order=order,
        )

    if error:
        st.error(error)
        return

    if not data or not data.get("items"):
        st.info("No reviews match the selected filters.")
        return

    df = pd.DataFrame(data["items"])
    df["review_date"] = pd.to_datetime(df["review_date"])

    if isinstance(date_from, date):
        df = df[df["review_date"].dt.date >= date_from]
    if isinstance(date_to, date):
        df = df[df["review_date"].dt.date <= date_to]

    total_filtered = len(df)
    if total_filtered == 0:
        st.info("No reviews match the selected filters (including the date range).")
        return

    st.caption(
        f"Showing results from the {min(data['total'], EXPLORER_FETCH_SIZE)} most recent matching "
        f"reviews returned by the API (API total: {data['total']}). Date filtering is applied to this batch."
    )

    total_pages = max(1, -(-total_filtered // EXPLORER_PAGE_SIZE))
    current_page = st.session_state.get("explorer_page", 1)
    current_page = min(max(current_page, 1), total_pages)

    start = (current_page - 1) * EXPLORER_PAGE_SIZE
    end = start + EXPLORER_PAGE_SIZE
    page_df = df.iloc[start:end]

    display_df = page_df[
        ["review_date", "user_name", "rating", "sentiment_label", "sentiment_score", "review_text", "app_version"]
    ].rename(
        columns={
            "review_date": "Date",
            "user_name": "User",
            "rating": "Rating",
            "sentiment_label": "Sentiment",
            "sentiment_score": "Score",
            "review_text": "Review",
            "app_version": "App Version",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=420)

    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("⬅ Previous", disabled=current_page <= 1, width="stretch"):
            st.session_state["explorer_page"] = current_page - 1
            st.rerun()
    with nav_col2:
        st.markdown(
            f"<div class='page-info'>Page {current_page} of {total_pages}"
            f" &nbsp;·&nbsp; {total_filtered:,} matching reviews</div>",
            unsafe_allow_html=True,
        )
    with nav_col3:
        if st.button("Next ➡", disabled=current_page >= total_pages, width="stretch"):
            st.session_state["explorer_page"] = current_page + 1
            st.rerun()


# ---------------------------------------------------------------------------
# Page: Data Quality & ETL Monitoring
# ---------------------------------------------------------------------------
def render_data_quality() -> None:
    st.markdown(
        '<div class="dashboard-header"><h1>Data Quality &amp; ETL Monitoring</h1>'
        '<p class="dashboard-subtitle">Pipeline health and run history, sourced from the etl_runs table.</p></div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading ETL statistics..."):
        etl_stats, error = fetch_etl_stats(history=20)

    if error:
        st.error(error)
        return

    if not etl_stats or not etl_stats.get("latest_run"):
        st.info("No ETL runs recorded yet. Run `python -m etl.load` to populate this page.")
        return

    latest = etl_stats["latest_run"]

    # Five equal columns compress badly on medium-width screens, especially
    # with the sidebar open. Two rows of 2-3 keeps each metric card readable.
    kpi_row1 = st.columns(3)
    kpi_row1[0].metric(" Rows Processed", f"{latest['final_row_count']:,}")
    kpi_row1[1].metric(" New Reviews Inserted", f"{latest['new_reviews_inserted']:,}")
    kpi_row1[2].metric(" Dupes Removed (Transform)", f"{latest['duplicates_removed']:,}")
    kpi_row2 = st.columns(2)
    kpi_row2[0].metric(" Dupes Skipped (Load)", f"{latest['duplicate_reviews_skipped']:,}")
    kpi_row2[1].metric(" Nulls Handled", f"{latest['nulls_handled']:,}")

    st.markdown("####")
    status_col, time_col = st.columns(2)
    with status_col:
        badge_class = "status-badge-success" if latest["status"] == "SUCCESS" else "status-badge-failed"
        badge_icon = "✅" if latest["status"] == "SUCCESS" else "❌"
        st.markdown(
            f"**Last ETL Status:** "
            f"<span class='{badge_class}'>{badge_icon} {latest['status']}</span>",
            unsafe_allow_html=True,
        )
        if latest["status"] != "SUCCESS" and latest.get("error_message"):
            st.error(f"Error: {latest['error_message']}")
    with time_col:
        st.markdown(f"**Last ETL Execution:** {format_relative_timestamp(latest['run_timestamp'])}")
        st.markdown(f"**App ID(s):** {latest['app_ids']}")

    st.markdown("####")
    st.plotly_chart(
        build_etl_run_history_chart(etl_stats.get("recent_runs", [])),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.markdown("#### Recent ETL Runs")
    runs_df = pd.DataFrame(etl_stats.get("recent_runs", []))
    if not runs_df.empty:
        runs_df["run_timestamp"] = pd.to_datetime(runs_df["run_timestamp"])
        display_cols = [
            "run_timestamp", "status", "raw_count", "duplicates_removed",
            "nulls_handled", "final_row_count", "new_reviews_inserted",
            "duplicate_reviews_skipped",
        ]
        st.dataframe(
            runs_df[display_cols].rename(
                columns={
                    "run_timestamp": "Run Time",
                    "status": "Status",
                    "raw_count": "Raw Count",
                    "duplicates_removed": "Dupes Removed (Transform)",
                    "nulls_handled": "Nulls Handled",
                    "final_row_count": "Final Row Count",
                    "new_reviews_inserted": "New Inserted",
                    "duplicate_reviews_skipped": "Dupes Skipped (Load)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Review Intelligence Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    page, app_id = render_sidebar()

    if page == "Executive Overview":
        render_executive_overview(app_id)
    elif page == "Sentiment Analytics":
        render_sentiment_analytics(app_id)
    elif page == "Keyword Intelligence":
        render_keyword_intelligence(app_id)
    elif page == "Review Explorer":
        render_review_explorer(app_id)
    elif page == "Data Quality & ETL Monitoring":
        render_data_quality()


if __name__ == "__main__":
    main()