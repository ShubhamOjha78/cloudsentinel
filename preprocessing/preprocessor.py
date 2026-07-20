"""
CloudSentinel – preprocessing/preprocessor.py
==================================================
Handles all data cleaning, normalization, and
sliding-window feature extraction for ML models.

Pipeline:
  1. Load metrics from DB (or DataFrame)
  2. Handle missing values (forward-fill)
  3. Min-Max normalize each feature column
  4. Build sliding windows for LSTM input
  5. Build flat feature vectors for Isolation Forest
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
FEATURE_COLS = [
    "cpu_percent",
    "mem_percent",
    "disk_read_mb",
    "disk_write_mb",
    "net_recv_kb",
    "net_sent_kb",
]

DEFAULT_WINDOW_SIZE  = 60   # 30 minutes at 30-second interval
SCALER_PATH          = "models/saved/scaler.pkl"


# ------------------------------------------------------------------
# Preprocessor class
# ------------------------------------------------------------------

class DataPreprocessor:
    """
    Stateful preprocessor that fits a MinMaxScaler on training data
    and applies the same scaler during inference.
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        self.window_size = window_size
        self.scaler: Optional[MinMaxScaler] = None

    # ---- Scaler persistence ----------------------------------------

    def fit_scaler(self, df: pd.DataFrame) -> None:
        """Fit MinMaxScaler on training data and save to disk."""
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(df[FEATURE_COLS])
        joblib.dump(self.scaler, SCALER_PATH)
        logger.info("Scaler fitted and saved to %s", SCALER_PATH)

    def load_scaler(self) -> None:
        """Load a previously fitted scaler from disk."""
        self.scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler loaded from %s", SCALER_PATH)

    # ---- Core pipeline steps ----------------------------------------

    @staticmethod
    def to_dataframe(records: List[dict]) -> pd.DataFrame:
        """Convert list of DB row dicts to a pandas DataFrame."""
        df = pd.DataFrame(records)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    @staticmethod
    def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values using forward-fill then backward-fill
        (backward-fill handles NaN at the start of the series).
        """
        missing_before = df[FEATURE_COLS].isnull().sum().sum()
        df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()
        missing_after = df[FEATURE_COLS].isnull().sum().sum()

        if missing_before > 0:
            logger.info(
                "Missing value imputation: %d → %d NaNs",
                missing_before, missing_after
            )
        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Min-Max normalization.
        Requires scaler to be fitted or loaded first.
        """
        if self.scaler is None:
            raise RuntimeError(
                "Scaler not fitted. Call fit_scaler() or load_scaler() first."
            )
        df = df.copy()
        df[FEATURE_COLS] = self.scaler.transform(df[FEATURE_COLS])
        return df

    # ---- Feature extraction ----------------------------------------

    def build_windows(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, List]:
        """
        Create sliding windows for LSTM input.

        Returns:
            X       – shape (n_windows, window_size, n_features)
            indices – list of ending-row indices for each window
        """
        data = df[FEATURE_COLS].values
        n    = len(data)

        if n < self.window_size:
            logger.warning(
                "Not enough rows (%d) for window size %d.",
                n, self.window_size
            )
            return np.empty((0, self.window_size, len(FEATURE_COLS))), []

        windows = []
        indices = []
        for i in range(self.window_size, n + 1):
            windows.append(data[i - self.window_size : i])
            indices.append(i - 1)

        return np.array(windows, dtype=np.float32), indices

    def build_flat_vectors(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return a 2-D array of shape (n_rows, n_features)
        for Isolation Forest input.
        """
        return df[FEATURE_COLS].values.astype(np.float32)

    # ---- Full pipeline ----------------------------------------

    def prepare_training_data(
        self,
        records: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        End-to-end preparation for model training.

        Returns:
            X_windows   – LSTM windows  (n, window_size, features)
            X_flat      – IF flat vecs  (n_rows, features)
            df_clean    – cleaned & normalized DataFrame
        """
        df = self.to_dataframe(records)
        df = self.handle_missing(df)
        self.fit_scaler(df)
        df = self.normalize(df)

        X_windows, _ = self.build_windows(df)
        X_flat        = self.build_flat_vectors(df)

        logger.info(
            "Training data prepared | rows=%d  windows=%d  flat=%d",
            len(df), len(X_windows), len(X_flat)
        )
        return X_windows, X_flat, df

    def prepare_inference_data(
        self,
        records: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        End-to-end preparation for model inference.
        Scaler must already be loaded.
        """
        df = self.to_dataframe(records)
        df = self.handle_missing(df)
        df = self.normalize(df)

        X_windows, _ = self.build_windows(df)
        X_flat        = self.build_flat_vectors(df)

        return X_windows, X_flat, df


# ------------------------------------------------------------------
# Module-level helper (used during detection loop)
# ------------------------------------------------------------------

def get_latest_window(records: List[dict], window_size: int = DEFAULT_WINDOW_SIZE):
    """
    Quick helper: returns the last window from a list of records.
    Used by the detection pipeline to get the most recent sequence.
    """
    preprocessor = DataPreprocessor(window_size=window_size)
    preprocessor.load_scaler()

    df = preprocessor.to_dataframe(records)
    df = preprocessor.handle_missing(df)
    df = preprocessor.normalize(df)

    data = df[FEATURE_COLS].values
    if len(data) < window_size:
        return None, None

    window = data[-window_size:].astype(np.float32)         # shape (window_size, features)
    flat   = data[-1:].astype(np.float32)                   # shape (1, features) — latest row
    return window, flat
