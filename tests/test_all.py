"""
CloudSentinel – tests/test_all.py
==================================================
Unit and integration tests using Python unittest.

Run:
    python -m pytest tests/ -v
    or
    python -m unittest tests.test_all -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ==================================================================
# Test 1: Preprocessor
# ==================================================================

class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        from preprocessing.preprocessor import DataPreprocessor, FEATURE_COLS
        self.DataPreprocessor = DataPreprocessor
        self.FEATURE_COLS     = FEATURE_COLS
        self.window_size = 5
        self.preprocessor = DataPreprocessor(window_size=self.window_size)

    def _make_df(self, n=20):
        """Create a dummy DataFrame with valid metric columns."""
        np.random.seed(42)
        data = {col: np.random.uniform(0, 100, n) for col in self.FEATURE_COLS}
        return pd.DataFrame(data)

    # UT-01: Forward-fill NaN values
    def test_handle_missing_fills_nan(self):
        df = self._make_df(10)
        df.loc[3, "cpu_percent"] = np.nan
        df.loc[7, "mem_percent"] = np.nan
        result = self.preprocessor.handle_missing(df)
        self.assertEqual(result[self.FEATURE_COLS].isnull().sum().sum(), 0,
                         "All NaN values should be filled.")

    # UT-02: Min-Max normalization range
    def test_normalize_range(self):
        df = self._make_df(50)
        df.loc[0, "cpu_percent"] = 0.0
        df.loc[1, "cpu_percent"] = 100.0

        self.preprocessor.fit_scaler(df)
        df_norm = self.preprocessor.normalize(df)

        for col in self.FEATURE_COLS:
            self.assertGreaterEqual(df_norm[col].min(), -1e-9,
                                    f"{col} min should be >= 0 after normalization")
            self.assertLessEqual(df_norm[col].max(), 1 + 1e-9,
                                  f"{col} max should be <= 1 after normalization")

    # UT-03: Sliding window shape
    def test_build_windows_shape(self):
        df = self._make_df(20)
        # Manually set scaler (use fit_scaler)
        self.preprocessor.fit_scaler(df)
        df_norm = self.preprocessor.normalize(df)
        windows, indices = self.preprocessor.build_windows(df_norm)
        expected_n = 20 - self.window_size + 1
        self.assertEqual(windows.shape, (expected_n, self.window_size, len(self.FEATURE_COLS)))

    # UT-04: Insufficient data returns empty windows
    def test_build_windows_insufficient_data(self):
        df = self._make_df(3)
        self.preprocessor.fit_scaler(df)
        df_norm = self.preprocessor.normalize(df)
        windows, indices = self.preprocessor.build_windows(df_norm)
        self.assertEqual(len(windows), 0)

    # UT-05: Flat vectors shape
    def test_build_flat_vectors_shape(self):
        df = self._make_df(10)
        flat = self.preprocessor.build_flat_vectors(df)
        self.assertEqual(flat.shape, (10, len(self.FEATURE_COLS)))


# ==================================================================
# Test 2: Isolation Forest
# ==================================================================

class TestIsolationForestDetector(unittest.TestCase):

    def setUp(self):
        from models.isolation_forest import IsolationForestDetector
        self.IsolationForestDetector = IsolationForestDetector
        self.detector = IsolationForestDetector(
            n_estimators=50, contamination=0.05, random_state=42
        )
        # Train on synthetic normal data
        np.random.seed(0)
        self.X_normal = np.random.normal(loc=0.3, scale=0.1, size=(200, 6)).clip(0, 1)
        self.X_anomaly = np.random.uniform(0.85, 1.0, size=(20, 6))
        self.detector.fit(self.X_normal)

    # UT-06: Normal data scores below threshold
    def test_normal_score_below_threshold(self):
        scores = self.detector.score(self.X_normal[:10])
        self.assertTrue(
            np.mean(scores) < 0.65,
            "Mean score for normal data should be below threshold"
        )

    # UT-07: Anomalous data scores above threshold
    def test_anomaly_score_above_threshold(self):
        scores = self.detector.score(self.X_anomaly)
        self.assertTrue(
            np.mean(scores) >= 0.50,
            "Mean score for anomalous data should be elevated"
        )

    # UT-08: Score range is [0, 1]
    def test_score_in_valid_range(self):
        all_data = np.vstack([self.X_normal, self.X_anomaly])
        scores = self.detector.score(all_data)
        self.assertTrue(np.all(scores >= 0.0), "All scores should be >= 0")
        self.assertTrue(np.all(scores <= 1.0), "All scores should be <= 1")

    # UT-09: Predict returns binary values
    def test_predict_binary_output(self):
        predictions = self.detector.predict(self.X_normal[:20])
        unique_vals = set(np.unique(predictions))
        self.assertTrue(unique_vals.issubset({0, 1}),
                        "Predictions should be binary (0 or 1)")

    # UT-10: Single sample scoring works
    def test_single_sample_scoring(self):
        single = self.X_normal[0]
        score = self.detector.score(single)
        self.assertEqual(score.shape, (1,))


# ==================================================================
# Test 3: Ensemble
# ==================================================================

class TestEnsembleDetector(unittest.TestCase):

    def setUp(self):
        from models.ensemble import EnsembleDetector, AnomalyResult
        self.EnsembleDetector = EnsembleDetector
        self.AnomalyResult    = AnomalyResult

    # UT-11: Ensemble score formula
    def test_combine_scores_formula(self):
        det = self.EnsembleDetector(if_weight=0.4, lstm_weight=0.6)
        score = det.combine_scores(if_score=0.8, lstm_score=0.9)
        expected = 0.4 * 0.8 + 0.6 * 0.9
        self.assertAlmostEqual(score, expected, places=6)

    # UT-12: Severity classification HIGH
    def test_severity_high(self):
        det = self.EnsembleDetector()
        self.assertEqual(det.classify_severity(0.92), "HIGH")

    # UT-13: Severity classification MEDIUM
    def test_severity_medium(self):
        det = self.EnsembleDetector()
        self.assertEqual(det.classify_severity(0.80), "MEDIUM")

    # UT-14: Severity classification LOW
    def test_severity_low(self):
        det = self.EnsembleDetector()
        self.assertEqual(det.classify_severity(0.68), "LOW")

    # UT-15: Severity classification NORMAL
    def test_severity_normal(self):
        det = self.EnsembleDetector()
        self.assertEqual(det.classify_severity(0.40), "NORMAL")

    # UT-16: Weights must sum to 1.0
    def test_invalid_weights_raise_assertion(self):
        with self.assertRaises(AssertionError):
            self.EnsembleDetector(if_weight=0.6, lstm_weight=0.6)


# ==================================================================
# Test 4: AlertEngine
# ==================================================================

class TestAlertEngine(unittest.TestCase):

    def setUp(self):
        from alerting.alert_engine import AlertEngine, COOLDOWN_SECS
        self.AlertEngine   = AlertEngine
        self.COOLDOWN_SECS = COOLDOWN_SECS

    # UT-17: No anomaly → evaluate returns None
    @patch("alerting.alert_engine.insert_anomaly_event")
    def test_no_anomaly_returns_none(self, mock_insert):
        engine = self.AlertEngine()
        result = engine.evaluate(
            instance_id="i-001", ensemble_score=0.40,
            if_score=0.30, lstm_score=0.45,
            severity="NORMAL", is_anomaly=False
        )
        self.assertIsNone(result)
        mock_insert.assert_not_called()

    # UT-18: Anomaly within cooldown is suppressed (no email)
    @patch("alerting.alert_engine.mark_alert_sent")
    @patch("alerting.alert_engine.insert_anomaly_event", return_value=42)
    @patch.object(
        __import__("alerting.alert_engine", fromlist=["AlertEngine"]).AlertEngine,
        "_send_email", return_value=True
    )
    def test_cooldown_suppresses_second_alert(self, mock_send, mock_insert, mock_mark):
        engine = self.AlertEngine()

        # First alert
        engine.evaluate(
            instance_id="i-002", ensemble_score=0.90,
            if_score=0.85, lstm_score=0.93,
            severity="HIGH", is_anomaly=True
        )

        # Second alert immediately (within cooldown)
        mock_send.reset_mock()
        engine.evaluate(
            instance_id="i-002", ensemble_score=0.91,
            if_score=0.86, lstm_score=0.94,
            severity="HIGH", is_anomaly=True
        )

        # Email should NOT be sent the second time
        mock_send.assert_not_called()

    # UT-19: SMTP credentials missing → email skipped gracefully
    @patch("alerting.alert_engine.insert_anomaly_event", return_value=99)
    @patch("alerting.alert_engine.mark_alert_sent")
    @patch("alerting.alert_engine.SMTP_USER", "")
    @patch("alerting.alert_engine.SMTP_PASSWORD", "")
    def test_missing_smtp_credentials_no_crash(self, mock_mark, mock_insert):
        engine = self.AlertEngine()
        result = engine.evaluate(
            instance_id="i-003", ensemble_score=0.88,
            if_score=0.80, lstm_score=0.92,
            severity="HIGH", is_anomaly=True
        )
        # Should still return event_id from DB insert
        self.assertEqual(result, 99)


# ==================================================================
# Test 5: DB Utils (mocked)
# ==================================================================

class TestDBUtils(unittest.TestCase):

    @patch("database.db_utils.get_connection")
    def test_insert_metric_calls_execute(self, mock_conn_ctx):
        from database.db_utils import insert_metric

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__  = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (1,)

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__  = MagicMock(return_value=False)
        mock_conn.cursor    = MagicMock(return_value=mock_cursor)
        mock_conn_ctx.return_value = mock_conn

        result = insert_metric("i-001", {
            "cpu_percent": 50.0, "mem_percent": 60.0,
            "disk_read_mb": 1.0, "disk_write_mb": 0.5,
            "net_recv_kb": 100.0, "net_sent_kb": 50.0,
        })
        self.assertEqual(result, 1)
        mock_cursor.execute.assert_called_once()


# ==================================================================
# Runner
# ==================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
