"""
CloudSentinel – models/ensemble.py
==================================================
Weighted ensemble that combines Isolation Forest
and LSTM anomaly scores into a single decision.

Score combination formula:
    ensemble_score = (w_if × if_score) + (w_lstm × lstm_score)

Severity thresholds:
    HIGH   : ensemble_score >= 0.88
    MEDIUM : 0.75 <= ensemble_score < 0.88
    LOW    : 0.65 <= ensemble_score < 0.75
    NORMAL : ensemble_score < 0.65
"""

import logging
from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np

from models.isolation_forest import IsolationForestDetector, load_isolation_forest
from models.lstm_autoencoder  import LSTMDetector, load_lstm_detector

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Default weights  (validated via grid search in training notebook)
# ------------------------------------------------------------------
IF_WEIGHT   = 0.40
LSTM_WEIGHT = 0.60

# ------------------------------------------------------------------
# Anomaly threshold & severity bands
# ------------------------------------------------------------------
ANOMALY_THRESHOLD  = 0.65

SEVERITY_HIGH      = 0.88
SEVERITY_MEDIUM    = 0.75
SEVERITY_LOW       = 0.65


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------

@dataclass
class AnomalyResult:
    is_anomaly:     bool
    severity:       str               # "NORMAL" | "LOW" | "MEDIUM" | "HIGH"
    ensemble_score: float
    if_score:       float
    lstm_score:     float
    details:        dict = field(default_factory=dict)


# ------------------------------------------------------------------
# EnsembleDetector
# ------------------------------------------------------------------

class EnsembleDetector:
    """
    Loads both detectors and combines their outputs.

    Usage:
        detector = EnsembleDetector()
        detector.load()
        result = detector.detect(window_sequence, flat_vector)
    """

    def __init__(
        self,
        if_weight:   float = IF_WEIGHT,
        lstm_weight: float = LSTM_WEIGHT,
        threshold:   float = ANOMALY_THRESHOLD,
    ):
        assert abs(if_weight + lstm_weight - 1.0) < 1e-6, \
            "IF and LSTM weights must sum to 1.0"

        self.if_weight   = if_weight
        self.lstm_weight = lstm_weight
        self.threshold   = threshold

        self._if_detector:   Optional[IsolationForestDetector] = None
        self._lstm_detector: Optional[LSTMDetector]            = None

    # ---- Lifecycle ------------------------------------------------

    def load(self) -> None:
        """Load both sub-detectors from disk."""
        self._if_detector   = load_isolation_forest()
        self._lstm_detector = load_lstm_detector()
        logger.info(
            "Ensemble loaded | w_IF=%.2f  w_LSTM=%.2f  threshold=%.2f",
            self.if_weight, self.lstm_weight, self.threshold
        )

    # ---- Core logic ------------------------------------------------

    def combine_scores(
        self,
        if_score:   float,
        lstm_score: float,
    ) -> float:
        """Weighted average of individual anomaly scores."""
        return (self.if_weight * if_score) + (self.lstm_weight * lstm_score)

    @staticmethod
    def classify_severity(score: float) -> str:
        """Map numeric ensemble score to severity label."""
        if score >= SEVERITY_HIGH:
            return "HIGH"
        if score >= SEVERITY_MEDIUM:
            return "MEDIUM"
        if score >= SEVERITY_LOW:
            return "LOW"
        return "NORMAL"

    # ---- Detection ------------------------------------------------

    def detect(
        self,
        window: np.ndarray,   # shape (window_size, n_features) – for LSTM
        flat:   np.ndarray,   # shape (n_features,)              – for IF
    ) -> AnomalyResult:
        """
        Run both detectors on a single observation and return
        an AnomalyResult with the combined decision.

        Parameters
        ----------
        window : np.ndarray
            The last `window_size` normalised time steps.
            Shape: (window_size, n_features)
        flat   : np.ndarray
            The latest normalised metric vector.
            Shape: (n_features,) or (1, n_features)
        """
        if self._if_detector is None or self._lstm_detector is None:
            raise RuntimeError("Detectors not loaded. Call load() first.")

        # Individual scores (both in [0, 1])
        if_score   = float(self._if_detector.score(flat)[0])
        lstm_score = float(
            self._lstm_detector.score(window[np.newaxis, ...])[0]
        )

        ensemble_score = self.combine_scores(if_score, lstm_score)
        severity       = self.classify_severity(ensemble_score)
        is_anomaly     = ensemble_score >= self.threshold

        result = AnomalyResult(
            is_anomaly     = is_anomaly,
            severity       = severity,
            ensemble_score = round(ensemble_score, 4),
            if_score       = round(if_score, 4),
            lstm_score     = round(lstm_score, 4),
            details        = {
                "if_weight":   self.if_weight,
                "lstm_weight": self.lstm_weight,
                "threshold":   self.threshold,
            },
        )

        if is_anomaly:
            logger.warning(
                "ANOMALY [%s] | ensemble=%.4f  IF=%.4f  LSTM=%.4f",
                severity, ensemble_score, if_score, lstm_score
            )
        else:
            logger.debug(
                "Normal  | ensemble=%.4f  IF=%.4f  LSTM=%.4f",
                ensemble_score, if_score, lstm_score
            )

        return result

    def batch_detect(
        self,
        windows: np.ndarray,  # (n, window_size, features)
        flats:   np.ndarray,  # (n, features)
    ) -> list:
        """Run detection on a batch of observations."""
        results = []
        for i in range(len(windows)):
            result = self.detect(windows[i], flats[i])
            results.append(result)
        return results
