"""
etl/db.py
==========
SQLAlchemy engine and session management, shared by the ETL pipeline and
the FastAPI service. Centralizing this avoids two different connection-pool
configurations drifting apart.

Design notes
------------
* `pool_pre_ping=True` issues a lightweight "SELECT 1" before handing out a
  pooled connection, so a connection silently dropped by Aiven (idle
  timeout, network blip) is detected and replaced instead of causing a
  confusing "MySQL server has gone away" error mid-request.
* `pool_recycle=280` proactively recycles connections before most managed
  MySQL providers' default idle/wait timeouts (commonly 300s), reducing
  how often `pool_pre_ping` even needs to kick in.
* The engine is created lazily (on first call to `get_engine()`) and
  cached, rather than at import time, so importing this module never
  requires a configured environment or a live database.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from etl.config import get_settings

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None


def get_engine() -> Engine:
    """Create (once) and return the SQLAlchemy engine for Aiven MySQL."""
    global _engine
    if _engine is None:
        settings = get_settings()
        logger.info(
            "Creating SQLAlchemy engine for host=%s db=%s",
            settings.aiven_host,
            settings.aiven_database,
        )
        _engine = create_engine(
            settings.database_url,
            connect_args=settings.ssl_connect_args,
            pool_pre_ping=True,
            pool_recycle=280,
            pool_size=5,
            max_overflow=5,
            future=True,
        )
    return _engine


def get_session_factory() -> sessionmaker:
    """Create (once) and return the session factory bound to the engine."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _SessionFactory


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Provide a transactional scope around a series of operations.

    Commits on success, rolls back and re-raises on any exception, and
    always closes the session.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        logger.exception("Transaction failed, rolling back.")
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session and guarantees it is
    closed after the request, regardless of outcome.
    """
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def check_connection() -> bool:
    """Run a trivial query to verify connectivity. Used by /health."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("Database health check failed.")
        return False