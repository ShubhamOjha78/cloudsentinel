"""
CloudSentinel – detection_loop.py
==================================================
Main real-time detection pipeline.

Runs continuously:
  1. Fetch latest `window_size` metric rows from DB
  2. Preprocess (normalize, build window + flat vector)
  3. Run ensemble detector
  4. If anomaly → alert engine
  5. Sleep for POLL_INTERVAL seconds → repeat

Usage:
    python detection_loop.py --instance-id i-0abc1234 --interval 30
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from database.db_utils           import fetch_metrics_window
from preprocessing.preprocessor  import DataPreprocessor, FEATURE_COLS
import numpy as np
from models.ensemble             import EnsembleDetector
from alerting.alert_engine       import AlertEngine

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] detection | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
DEFAULT_INTERVAL    = int(os.getenv("DETECT_INTERVAL", 30))
DEFAULT_WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 60))
DEFAULT_INSTANCE    = os.getenv("INSTANCE_ID", "local-dev-01")


def run_detection_loop(
    instance_id:  str,
    poll_interval: int = DEFAULT_INTERVAL,
    window_size:   int = DEFAULT_WINDOW_SIZE,
) -> None:
    """
    Infinite detection loop for a single cloud instance.
    """
    logger.info(
        "Starting detection loop | instance=%s  poll=%ds  window=%d",
        instance_id, poll_interval, window_size
    )

    # Load models once at startup
    ensemble = EnsembleDetector()
    ensemble.load()

    alert_engine = AlertEngine()
    
    preprocessor = DataPreprocessor(window_size=window_size)
    preprocessor.load_scaler()

    while True:
        try:
            # 1. Fetch latest window of raw metric records
            records = fetch_metrics_window(instance_id, window_size=window_size)

            if len(records) < window_size:
                logger.info(
                    "Waiting for data… (%d/%d rows available)",
                    len(records), window_size
                )
                time.sleep(poll_interval)
                continue

            # 2. Preprocess: normalize + build sequences
            
            df = preprocessor.to_dataframe(records)
            df = preprocessor.handle_missing(df)
            df = preprocessor.normalize(df)

            if len(df) < window_size:
                logger.info("Not enough data yet (%d/%d)", len(df), window_size)
                time.sleep(poll_interval)
                continue

            
            data   = df[FEATURE_COLS].values.astype("float32")
            window = data[-window_size:]
            flat = data[-1:]

            # 3. Ensemble detection
            result = ensemble.detect(window, flat)

            # 4. Alert if anomaly detected
            if result.is_anomaly:
                alert_engine.evaluate(
                    instance_id    = instance_id,
                    ensemble_score = result.ensemble_score,
                    if_score       = result.if_score,
                    lstm_score     = result.lstm_score,
                    severity       = result.severity,
                    is_anomaly     = result.is_anomaly,
                )
            else:
                logger.debug(
                    "Status: NORMAL | score=%.4f  IF=%.4f  LSTM=%.4f",
                    result.ensemble_score, result.if_score, result.lstm_score
                )

        except KeyboardInterrupt:
            logger.info("Detection loop stopped by user.")
            break
        except Exception as exc:
            logger.error("Unexpected error in detection loop: %s", exc, exc_info=True)

        time.sleep(poll_interval)


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CloudSentinel – Real-Time Detection Loop"
    )
    parser.add_argument(
        "--instance-id", default=DEFAULT_INSTANCE,
        help="Cloud instance ID to monitor (default: %(default)s)"
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help="Detection poll interval in seconds (default: %(default)s)"
    )
    parser.add_argument(
        "--window-size", type=int, default=DEFAULT_WINDOW_SIZE,
        help="Sliding window size (default: %(default)s)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_detection_loop(
        instance_id   = args.instance_id,
        poll_interval = args.interval,
        window_size   = args.window_size,
    )
