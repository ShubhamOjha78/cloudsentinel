"""
CloudSentinel – train_all.py
==================================================
One-shot script to train both models on historical
normal-period data fetched from the DB.

Usage:
    python train_all.py --instance-id i-0abc1234 --days 30
"""

import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from database.db_utils          import fetch_metrics_range
from preprocessing.preprocessor import DataPreprocessor
from models.isolation_forest    import IsolationForestDetector
from models.lstm_autoencoder    import LSTMDetector

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] train | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train(instance_id: str, days: int = 30, window_size: int = 60) -> None:
    logger.info(
        "Starting training | instance=%s  days=%d  window_size=%d",
        instance_id, days, window_size
    )

    # ---- 1. Fetch historical data ----------------------------------
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    logger.info("Fetching metrics from %s to %s …", start.date(), end.date())

    records = fetch_metrics_range(instance_id, start, end)
    logger.info("Fetched %d metric records.", len(records))

    if len(records) < window_size * 2:
        logger.error(
            "Insufficient data: need at least %d records, got %d.",
            window_size * 2, len(records)
        )
        sys.exit(1)

    # ---- 2. Preprocess ---------------------------------------------
    preprocessor = DataPreprocessor(window_size=window_size)
    X_windows, X_flat, df_clean = preprocessor.prepare_training_data(records)

    logger.info(
        "Preprocessing done | rows=%d  windows=%d  flat_shape=%s",
        len(df_clean), len(X_windows), X_flat.shape
    )

    # ---- 3. Train Isolation Forest ----------------------------------
    logger.info("--- Training Isolation Forest ---")
    if_detector = IsolationForestDetector(
        n_estimators  = 200,
        contamination = 0.05,
        random_state  = 42,
    )
    if_detector.fit(X_flat)

    # ---- 4. Train LSTM Autoencoder ----------------------------------
    logger.info("--- Training LSTM Autoencoder ---")
    lstm_detector = LSTMDetector(
        window_size   = window_size,
        n_features    = 6,
        encoder_units = (128, 64),
        decoder_units = (64, 128),
        learning_rate = 0.001,
        epochs        = 50,
        batch_size    = 64,
        patience      = 5,
        val_split     = 0.20,
    )
    history = lstm_detector.fit(X_windows)

    # ---- 5. Summary --------------------------------------------------
    final_train_loss = history["loss"][-1]
    final_val_loss   = history.get("val_loss", [None])[-1]

    logger.info(
        "Training complete!\n"
        "  Isolation Forest: saved to models/saved/isolation_forest.pkl\n"
        "  LSTM Autoencoder: saved to models/saved/lstm_autoencoder.h5\n"
        "  Final train loss: %.6f\n"
        "  Final val loss:   %s",
        final_train_loss,
        f"{final_val_loss:.6f}" if final_val_loss else "N/A"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="CloudSentinel – Train All Models"
    )
    parser.add_argument(
        "--instance-id", required=True,
        help="Instance ID to use as training data source"
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Number of historical days to use (default: 30)"
    )
    parser.add_argument(
        "--window-size", type=int, default=60,
        help="Sliding window size (default: 60)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        instance_id = args.instance_id,
        days        = args.days,
        window_size = args.window_size,
    )
