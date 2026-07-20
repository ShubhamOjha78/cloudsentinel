"""
CloudSentinel – models/isolation_forest.py
==================================================
Isolation Forest anomaly detector.

Key design decisions:
  - Trained on NORMAL-period data only.
  - decision_function() returns higher score for normal points.
    We invert and normalize to [0,1] so that score=1 means most anomalous.
  - contamination=0.05  →  ~5% of training data assumed noisy/borderline.
"""

import logging
import os
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
SAVE_DIR   = Path("models/saved")
MODEL_PATH = SAVE_DIR / "isolation_forest.pkl"

SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# IsolationForestDetector
# ------------------------------------------------------------------

class IsolationForestDetector:
    """
    Wrapper around sklearn IsolationForest.

    Anomaly score convention (CloudSentinel):
        0.0  →  very normal
        1.0  →  very anomalous
    """

    def __init__(
        self,
        n_estimators: int   = 200,
        contamination: float = 0.05,
        max_samples: Union[float, Literal["auto"]] = "auto",
        random_state: int   = 42,
    ):
        self.n_estimators   = n_estimators
        self.contamination  = contamination
        self.max_samples: Union[float, Literal["auto"]] = max_samples
        self.random_state   = random_state
        self.model: Optional[IsolationForest] = None

        # Calibration stats (computed after fitting)
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    # ---- Training ------------------------------------------------

    def fit(self, X_train: np.ndarray) -> None:
        """
        Fit the Isolation Forest on normal-period flat vectors.
        X_train shape: (n_samples, n_features)
        """
        logger.info(
            "Training Isolation Forest | samples=%d  features=%d  "
            "n_estimators=%d  contamination=%.2f",
            X_train.shape[0], X_train.shape[1],
            self.n_estimators, self.contamination
        )
        self.model = IsolationForest(
            n_estimators  = self.n_estimators,
            contamination = self.contamination,
            max_samples   = self.max_samples,
            random_state  = self.random_state,
            n_jobs        = -1,
        )
        self.model.fit(X_train)

        # Calibrate score range on training data for normalization
        raw_scores       = self.model.decision_function(X_train)
        self._score_min  = float(raw_scores.min())
        self._score_max  = float(raw_scores.max())

        logger.info(
            "Training complete | raw score range: [%.4f, %.4f]",
            self._score_min, self._score_max
        )
        self.save()

    # ---- Scoring ------------------------------------------------

    def _normalize_score(self, raw: np.ndarray) -> np.ndarray:
        """
        Invert sklearn score (higher = normal) and min-max normalize
        to CloudSentinel convention (higher = more anomalous, 0-1).
        """
        # Invert: more negative sklearn score → higher anomaly risk
        inverted = -raw
        # Shift and scale using training calibration values
        denom = max((-self._score_min) - (-self._score_max), 1e-9)
        normed = (inverted - (-self._score_max)) / denom
        return np.clip(normed, 0.0, 1.0)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return normalized anomaly scores in [0, 1].
        X shape: (n_samples, n_features) or (n_features,) for single point.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        X = np.atleast_2d(X)
        raw    = self.model.decision_function(X)
        scores = self._normalize_score(raw)
        return scores

    def predict(self, X: np.ndarray, threshold: float = 0.65) -> np.ndarray:
        """
        Classify samples as anomaly (1) or normal (0).
        Uses normalized anomaly score compared to threshold.
        """
        scores = self.score(X)
        return (scores >= threshold).astype(int)

    # ---- Evaluation ------------------------------------------------

    def evaluate(
        self,
        X_test: np.ndarray,
        y_true: np.ndarray,
        threshold: float = 0.65
    ) -> dict:
        """
        Evaluate on a labelled test set.
        y_true: binary array, 1 = anomaly.
        """
        y_pred = self.predict(X_test, threshold=threshold)
        scores = self.score(X_test)

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        metrics = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
        }
        logger.info("IF Evaluation | P=%.4f R=%.4f F1=%.4f",
                    metrics["precision"], metrics["recall"], metrics["f1"])
        print(classification_report(y_true, y_pred,
                                    target_names=["Normal", "Anomaly"]))
        return metrics

    # ---- Persistence ------------------------------------------------

    def save(self, path: Path = MODEL_PATH) -> None:
        payload = {
            "model":       self.model,
            "score_min":   self._score_min,
            "score_max":   self._score_max,
            "n_estimators":self.n_estimators,
            "contamination":self.contamination,
        }
        joblib.dump(payload, path)
        logger.info("Isolation Forest saved → %s", path)

    def load(self, path: Path = MODEL_PATH) -> None:
        payload          = joblib.load(path)
        self.model       = payload["model"]
        self._score_min  = payload["score_min"]
        self._score_max  = payload["score_max"]
        logger.info("Isolation Forest loaded ← %s", path)


# ------------------------------------------------------------------
# Convenience factory
# ------------------------------------------------------------------

def load_isolation_forest() -> IsolationForestDetector:
    """Load a saved Isolation Forest detector."""
    detector = IsolationForestDetector()
    detector.load()
    return detector
