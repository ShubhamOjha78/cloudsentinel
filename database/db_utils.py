"""
CloudSentinel – database/db_utils.py
Database connection and CRUD utilities for PostgreSQL.
"""

import os
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Load .env from project root OR parent directory (Windows-friendly)
from pathlib import Path
_env_candidates = [
    Path(__file__).resolve().parent.parent / "config.env",
    Path(__file__).resolve().parent.parent / ".env",
    Path("config.env"),
    Path(".env"),
]
for _env_path in _env_candidates:
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path)
        break

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Connection configuration (loaded from .env / environment)
# ------------------------------------------------------------------
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "cloudsentinel"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}
# Only add sslmode if not connecting to localhost (avoids Windows SSL errors)
_sslmode = os.getenv("DB_SSLMODE", "")
if _sslmode:
    DB_CONFIG["sslmode"] = _sslmode


def check_db_connection() -> tuple[bool, str]:
    """
    Test whether the database is reachable.
    Returns (True, "") on success or (False, error_message) on failure.
    """
    try:
        conn = psycopg2.connect(connect_timeout=3, **DB_CONFIG)
        conn.close()
        return True, ""
    except psycopg2.OperationalError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)


@contextmanager
def get_connection():
    """Context manager that yields a psycopg2 connection."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
        conn.commit()
    except psycopg2.OperationalError as exc:
        if conn:
            conn.rollback()
        logger.error("DB connection failed: %s", exc)
        raise ConnectionError(
            f"Cannot connect to PostgreSQL at "
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}.\n"
            f"Original error: {exc}"
        ) from exc
    except Exception as exc:
        if conn:
            conn.rollback()
        logger.error("Database error: %s", exc)
        raise
    finally:
        if conn:
            conn.close()


# ------------------------------------------------------------------
# Metrics CRUD
# ------------------------------------------------------------------

def insert_metric(instance_id: str, metrics: Dict[str, float]) -> int:
    """
    Insert one row into cloud_metrics.
    Returns the new metric_id.

    metrics dict keys:
        cpu_percent, mem_percent,
        disk_read_mb, disk_write_mb,
        net_recv_kb, net_sent_kb
    """
    sql = """
        INSERT INTO cloud_metrics
            (instance_id, cpu_percent, mem_percent,
             disk_read_mb, disk_write_mb, net_recv_kb, net_sent_kb)
        VALUES
            (%(instance_id)s, %(cpu_percent)s, %(mem_percent)s,
             %(disk_read_mb)s, %(disk_write_mb)s,
             %(net_recv_kb)s, %(net_sent_kb)s)
        RETURNING metric_id
    """
    params = {"instance_id": instance_id, **metrics}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()[0]


def fetch_recent_metrics(
    instance_id: str,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Fetch the most recent `limit` metric rows for an instance."""
    sql = """
        SELECT metric_id, instance_id, timestamp,
               cpu_percent, mem_percent,
               disk_read_mb, disk_write_mb,
               net_recv_kb, net_sent_kb
        FROM cloud_metrics
        WHERE instance_id = %s
        ORDER BY timestamp DESC
        LIMIT %s
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (instance_id, limit))
            rows = cur.fetchall()
    # Return in chronological order
    return list(reversed([dict(r) for r in rows]))


def fetch_metrics_window(
    instance_id: str,
    window_size: int = 60
) -> List[Dict[str, Any]]:
    """Fetch the last `window_size` rows – used for model inference."""
    return fetch_recent_metrics(instance_id, limit=window_size)


def fetch_metrics_range(
    instance_id: str,
    start: datetime,
    end: datetime
) -> List[Dict[str, Any]]:
    """Fetch metrics for a given time range."""
    sql = """
        SELECT * FROM cloud_metrics
        WHERE instance_id = %s
          AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp ASC
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (instance_id, start, end))
            return [dict(r) for r in cur.fetchall()]


# ------------------------------------------------------------------
# Anomaly events CRUD
# ------------------------------------------------------------------

def insert_anomaly_event(
    instance_id: str,
    if_score: float,
    lstm_error: float,
    ensemble_score: float,
    severity: str,
    notes: str = ""
) -> int:
    """Insert a detected anomaly event. Returns new event_id."""
    sql = """
        INSERT INTO anomaly_events
            (instance_id, if_score, lstm_error,
             ensemble_score, severity, notes)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING event_id
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                instance_id, if_score, lstm_error,
                ensemble_score, severity, notes
            ))
            return cur.fetchone()[0]


def mark_alert_sent(event_id: int) -> None:
    """Mark an anomaly event's alert_sent flag as TRUE."""
    sql = "UPDATE anomaly_events SET alert_sent = TRUE WHERE event_id = %s"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (event_id,))


def fetch_recent_anomalies(
    instance_id: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Fetch recent anomaly events, optionally filtered by instance."""
    if instance_id:
        sql = """
            SELECT * FROM anomaly_events
            WHERE instance_id = %s
            ORDER BY detected_at DESC LIMIT %s
        """
        params = (instance_id, limit)
    else:
        sql = """
            SELECT * FROM anomaly_events
            ORDER BY detected_at DESC LIMIT %s
        """
        params = (limit,)

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]


def get_distinct_instances() -> List[str]:
    """Return list of all monitored instance IDs."""
    sql = "SELECT DISTINCT instance_id FROM cloud_metrics ORDER BY instance_id"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [row[0] for row in cur.fetchall()]
