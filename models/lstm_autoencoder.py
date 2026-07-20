"""
CloudSentinel – models/lstm_autoencoder.py
==================================================
LSTM Autoencoder for time-series anomaly detection.

Architecture (encoder-decoder):
  Input (60 steps × 6 features)
    → Encoder LSTM 128 units
    → Encoder LSTM  64 units
    → RepeatVector(60)
    → Decoder LSTM  64 units
    → Decoder LSTM 128 units
    → TimeDistributed Dense(6)
  Output (60 steps × 6 features)

Anomaly scoring:
  reconstruction_error = mean(MSE over all time steps and features)
  Normalized to [0,1] using training-set calibration stats.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import joblib
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Lazy TF import (keeps startup fast when TF not needed)
# ------------------------------------------------------------------
def _import_tf():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model, load_model
        from tensorflow.keras.layers import (
            Input, LSTM, Dense, RepeatVector, TimeDistributed
        )
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        return tf, Model, load_model, Input, LSTM, Dense, RepeatVector, TimeDistributed, EarlyStopping, ModelCheckpoint
    except ImportError as exc:
        raise ImportError(
            "TensorFlow not installed. Run: pip install tensorflow"
        ) from exc


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
SAVE_DIR    = Path("models/saved")
MODEL_H5    = SAVE_DIR / "lstm_autoencoder.h5"
STATS_PATH  = SAVE_DIR / "lstm_stats.pkl"

SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# LSTMDetector
# ------------------------------------------------------------------

class LSTMDetector:
    """
    LSTM Autoencoder anomaly detector.

    Anomaly score convention (CloudSentinel):
        0.0  →  very normal   (low reconstruction error)
        1.0  →  very anomalous (high reconstruction error)
    """

    def __init__(
        self,
        window_size:   int   = 60,
        n_features:    int   = 6,
        encoder_units: tuple = (128, 64),
        decoder_units: tuple = (64, 128),
        learning_rate: float = 0.001,
        epochs:        int   = 50,
        batch_size:    int   = 64,
        patience:      int   = 5,
        val_split:     float = 0.20,
    ):
        self.window_size   = window_size
        self.n_features    = n_features
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.patience      = patience
        self.val_split     = val_split

        self.model: Optional[object] = None

        # Calibration stats
        self._threshold_mean: float = 0.0
        self._threshold_std:  float = 1.0
        self._error_min:      float = 0.0
        self._error_max:      float = 1.0

    # ---- Build model ------------------------------------------------

    def build_model(self):
        """Construct the encoder-decoder LSTM autoencoder."""
        tf, Model, _, Input, LSTM, Dense, RepeatVector, TimeDistributed, _, _ = _import_tf()

        inp = Input(shape=(self.window_size, self.n_features), name="input")

        # Encoder
        x = LSTM(self.encoder_units[0], return_sequences=True,
                 name="enc_lstm_1")(inp)
        x = LSTM(self.encoder_units[1], return_sequences=False,
                 name="enc_lstm_2")(x)

        # Bottleneck → repeat for decoder
        x = RepeatVector(self.window_size, name="repeat")(x)

        # Decoder
        x = LSTM(self.decoder_units[0], return_sequences=True,
                 name="dec_lstm_1")(x)
        x = LSTM(self.decoder_units[1], return_sequences=True,
                 name="dec_lstm_2")(x)

        # Output reconstruction
        out = TimeDistributed(Dense(self.n_features), name="output")(x)

        self.model = Model(inputs=inp, outputs=out, name="lstm_autoencoder")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )
        self.model.summary(print_fn=logger.info)
        return self.model

    # ---- Training ------------------------------------------------

    def fit(self, X_train: np.ndarray) -> dict:
        """
        Train the autoencoder on NORMAL-period windows.
        X_train shape: (n_samples, window_size, n_features)

        Returns training history dict.
        """
        _, _, _, _, _, _, _, _, EarlyStopping, ModelCheckpoint = _import_tf()

        if self.model is None:
            self.build_model()

        logger.info(
            "Training LSTM Autoencoder | samples=%d  window=%d  features=%d",
            X_train.shape[0], self.window_size, self.n_features
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(MODEL_H5),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ]

        history = self.model.fit(
            X_train, X_train,                # target = input (reconstruction)
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.val_split,
            callbacks=callbacks,
            shuffle=True,
            verbose=1,
        )

        # Calibrate error range on training data
        train_errors = self._compute_errors(X_train)
        self._threshold_mean = float(np.mean(train_errors))
        self._threshold_std  = float(np.std(train_errors))
        self._error_min      = float(train_errors.min())
        self._error_max      = float(train_errors.max())

        logger.info(
            "Training complete | mean_error=%.6f  std=%.6f  "
            "error_range=[%.6f, %.6f]",
            self._threshold_mean, self._threshold_std,
            self._error_min, self._error_max
        )
        self._save_stats()
        return history.history

    # ---- Scoring ------------------------------------------------

    def _compute_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample mean squared reconstruction error."""
        X_pred   = self.model.predict(X, verbose=0)
        # MSE averaged over (window_size × n_features)
        errors   = np.mean(np.square(X - X_pred), axis=(1, 2))
        return errors

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return normalized anomaly scores in [0, 1].
        X shape: (n_samples, window_size, n_features)
                 or (window_size, n_features) for a single window.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        X = np.atleast_3d(X)
        if X.ndim == 2:
            X = X[np.newaxis, ...]      # add batch dimension

        errors = self._compute_errors(X)
        denom  = max(self._error_max - self._error_min, 1e-9)
        normed = (errors - self._error_min) / denom
        return np.clip(normed, 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.65) -> np.ndarray:
        """Classify as anomaly (1) or normal (0)."""
        return (self.score(X) >= threshold).astype(int)

    # ---- Evaluation ------------------------------------------------

    def evaluate(
        self,
        X_test: np.ndarray,
        y_true: np.ndarray,
        threshold: float = 0.65,
    ) -> dict:
        y_pred = self.predict(X_test, threshold=threshold)
        metrics = {
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        }
        logger.info("LSTM Evaluation | P=%.4f R=%.4f F1=%.4f",
                    metrics["precision"], metrics["recall"], metrics["f1"])
        print(classification_report(y_true, y_pred,
                                    target_names=["Normal", "Anomaly"]))
        return metrics

    # ---- Persistence ------------------------------------------------

    def _save_stats(self) -> None:
        stats = {
            "threshold_mean": self._threshold_mean,
            "threshold_std":  self._threshold_std,
            "error_min":      self._error_min,
            "error_max":      self._error_max,
        }
        joblib.dump(stats, STATS_PATH)

    def save(self) -> None:
        self.model.save(str(MODEL_H5))
        self._save_stats()
        logger.info("LSTM model saved → %s", MODEL_H5)

    def load(self) -> None:
        _, _, load_model, *_ = _import_tf()
        self.model = load_model(str(MODEL_H5), compile=False)
        stats = joblib.load(STATS_PATH)
        self._threshold_mean = stats["threshold_mean"]
        self._threshold_std  = stats["threshold_std"]
        self._error_min      = stats["error_min"]
        self._error_max      = stats["error_max"]
        logger.info("LSTM model loaded ← %s", MODEL_H5)


# ------------------------------------------------------------------
# Convenience factory
# ------------------------------------------------------------------

def load_lstm_detector() -> LSTMDetector:
    detector = LSTMDetector()
    detector.load()
    return detector
