"""
CloudSentinel – agents/collector.py
==================================================
Monitoring agent that runs on each cloud instance.
Collects CPU, memory, disk, and network metrics
every INTERVAL seconds and posts them to the
central PostgreSQL database via db_utils.

Usage:
    python agents/collector.py --instance-id i-0abc1234 --interval 30

Dependencies: psutil, python-dotenv
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.db_utils import insert_metric

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] collector | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
DEFAULT_INTERVAL   = int(os.getenv("COLLECT_INTERVAL", 30))   # seconds
DEFAULT_INSTANCE   = os.getenv("INSTANCE_ID", "local-dev-01")
MAX_RETRIES        = 5
BACKOFF_BASE       = 2   # exponential back-off: 2, 4, 8, 16, 32 seconds


# ------------------------------------------------------------------
# Metric collection
# ------------------------------------------------------------------

class MetricsCollector:
    """
    Collects six system metrics using psutil.
    Disk/network values are computed as per-second deltas.
    """

    def __init__(self, instance_id: str, interval: int):
        self.instance_id = instance_id
        self.interval    = interval

        # Seed previous readings for delta computation
        self._prev_disk = psutil.disk_io_counters()
        self._prev_net  = psutil.net_io_counters()
        self._prev_time = time.monotonic()

        logger.info(
            "MetricsCollector initialised | instance=%s interval=%ds",
            instance_id, interval
        )

    def _read(self) -> dict:
        """Read raw psutil counters and compute per-second rates."""
        now_disk = psutil.disk_io_counters()
        now_net  = psutil.net_io_counters()
        now_time = time.monotonic()
        elapsed  = max(now_time - self._prev_time, 0.001)   # avoid div/0

        # Bytes → MB/s  |  Bytes → KB/s
        disk_read_mb  = (now_disk.read_bytes  - self._prev_disk.read_bytes)  / elapsed / 1_048_576
        disk_write_mb = (now_disk.write_bytes - self._prev_disk.write_bytes) / elapsed / 1_048_576
        net_recv_kb   = (now_net.bytes_recv   - self._prev_net.bytes_recv)   / elapsed / 1_024
        net_sent_kb   = (now_net.bytes_sent   - self._prev_net.bytes_sent)   / elapsed / 1_024

        # Update previous readings
        self._prev_disk = now_disk
        self._prev_net  = now_net
        self._prev_time = now_time

        return {
            "cpu_percent":  psutil.cpu_percent(interval=None),
            "mem_percent":  psutil.virtual_memory().percent,
            "disk_read_mb": round(max(disk_read_mb, 0), 4),
            "disk_write_mb":round(max(disk_write_mb, 0), 4),
            "net_recv_kb":  round(max(net_recv_kb, 0), 4),
            "net_sent_kb":  round(max(net_sent_kb, 0), 4),
        }

    def _store_with_retry(self, metrics: dict) -> None:
        """Push metrics to DB with exponential back-off on failure."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                metric_id = insert_metric(self.instance_id, metrics)
                logger.debug("Stored metric_id=%d | %s", metric_id, metrics)
                return
            except Exception as exc:
                wait = BACKOFF_BASE ** attempt
                logger.warning(
                    "DB insert failed (attempt %d/%d): %s — retrying in %ds",
                    attempt, MAX_RETRIES, exc, wait
                )
                time.sleep(wait)
        logger.error("All %d DB insert attempts failed — data point lost.", MAX_RETRIES)

    def run(self) -> None:
        """
        Main collection loop.
        Collects metrics every `self.interval` seconds indefinitely.
        """
        logger.info("Starting collection loop (Ctrl-C to stop) …")
        # Warm-up: first psutil.cpu_percent call returns 0.0
        psutil.cpu_percent(interval=None)
        time.sleep(self.interval)

        while True:
            try:
                metrics = self._read()
                logger.info(
                    "cpu=%.1f%%  mem=%.1f%%  disk_r=%.2fMB/s  disk_w=%.2fMB/s  "
                    "net_r=%.2fKB/s  net_s=%.2fKB/s",
                    metrics["cpu_percent"],
                    metrics["mem_percent"],
                    metrics["disk_read_mb"],
                    metrics["disk_write_mb"],
                    metrics["net_recv_kb"],
                    metrics["net_sent_kb"],
                )
                self._store_with_retry(metrics)
            except KeyboardInterrupt:
                logger.info("Collector stopped by user.")
                break
            except Exception as exc:
                logger.error("Unexpected error in collection loop: %s", exc)

            time.sleep(self.interval)


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CloudSentinel – Cloud Metrics Collector"
    )
    parser.add_argument(
        "--instance-id", default=DEFAULT_INSTANCE,
        help="Unique identifier for this cloud instance (default: %(default)s)"
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help="Collection interval in seconds (default: %(default)s)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collector = MetricsCollector(
        instance_id=args.instance_id,
        interval=args.interval,
    )
    collector.run()
