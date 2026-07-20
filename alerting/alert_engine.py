"""
CloudSentinel – alerting/alert_engine.py
==================================================
Alert engine that evaluates ensemble results and
dispatches email notifications with cooldown logic.

Flow:
  1. Receive AnomalyResult from EnsembleDetector
  2. If anomaly → insert into anomaly_events table
  3. Check cooldown (5-minute suppression window)
  4. If not suppressed → send HTML email via SMTP
  5. Mark alert_sent = TRUE in DB
"""

import logging
import os
import smtplib
import time
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# SMTP configuration (from .env)
# ------------------------------------------------------------------
SMTP_HOST      = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT", 587))
SMTP_USER      = os.getenv("SMTP_USER", "")
SMTP_PASSWORD  = os.getenv("SMTP_PASSWORD", "")
ALERT_FROM     = os.getenv("ALERT_FROM", SMTP_USER)
ALERT_TO       = os.getenv("ALERT_TO", "")           # comma-separated emails

# Cooldown: suppress repeat alerts for the same instance within N seconds
COOLDOWN_SECS  = int(os.getenv("ALERT_COOLDOWN_SECS", 300))  # 5 minutes


# ------------------------------------------------------------------
# HTML email template
# ------------------------------------------------------------------
EMAIL_TEMPLATE = """
<html><body style="font-family: Arial, sans-serif; color: #222;">
  <div style="background:#1F3864; padding:16px; border-radius:6px;">
    <h2 style="color:#fff; margin:0;">
      ⚠️ CloudSentinel Alert — {severity} Anomaly Detected
    </h2>
  </div>
  <div style="padding:16px; background:#F5F8FF; border-left:4px solid {color};">
    <table style="border-collapse:collapse; width:100%;">
      <tr><td style="padding:6px; font-weight:bold;">Instance ID</td>
          <td style="padding:6px;">{instance_id}</td></tr>
      <tr style="background:#EAF1FB;">
          <td style="padding:6px; font-weight:bold;">Detected At</td>
          <td style="padding:6px;">{detected_at}</td></tr>
      <tr><td style="padding:6px; font-weight:bold;">Severity</td>
          <td style="padding:6px; color:{color}; font-weight:bold;">{severity}</td></tr>
      <tr style="background:#EAF1FB;">
          <td style="padding:6px; font-weight:bold;">Ensemble Score</td>
          <td style="padding:6px;">{ensemble_score:.4f}</td></tr>
      <tr><td style="padding:6px; font-weight:bold;">Isolation Forest Score</td>
          <td style="padding:6px;">{if_score:.4f}</td></tr>
      <tr style="background:#EAF1FB;">
          <td style="padding:6px; font-weight:bold;">LSTM Error Score</td>
          <td style="padding:6px;">{lstm_score:.4f}</td></tr>
    </table>
  </div>
  <p style="padding:0 16px; color:#555; font-size:13px;">
    Please review the CloudSentinel dashboard for detailed metrics and anomaly history.
  </p>
  <p style="padding:0 16px; color:#999; font-size:11px;">
    This is an automated alert from CloudSentinel Real-Time Anomaly Detection System.<br>
    MCA (AI/ML) Project | Chandigarh University
  </p>
</body></html>
"""

SEVERITY_COLORS = {
    "HIGH":   "#C0392B",
    "MEDIUM": "#E67E22",
    "LOW":    "#F1C40F",
    "NORMAL": "#27AE60",
}


# ------------------------------------------------------------------
# AlertEngine
# ------------------------------------------------------------------

class AlertEngine:
    """
    Evaluates anomaly results, stores events in DB,
    and sends email notifications with cooldown suppression.
    """

    def __init__(self):
        # instance_id → last alert timestamp
        self._cooldown_map: Dict[str, datetime] = {}

    # ---- Cooldown logic ----------------------------------------

    def _is_in_cooldown(self, instance_id: str) -> bool:
        """Return True if an alert for this instance was recently sent."""
        last = self._cooldown_map.get(instance_id)
        if last is None:
            return False
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return elapsed < COOLDOWN_SECS

    def _record_alert(self, instance_id: str) -> None:
        self._cooldown_map[instance_id] = datetime.now(timezone.utc)

    # ---- Email dispatch ----------------------------------------

    def _build_email(
        self,
        instance_id:    str,
        detected_at:    datetime,
        ensemble_score: float,
        if_score:       float,
        lstm_score:     float,
        severity:       str,
    ) -> MIMEMultipart:
        msg           = MIMEMultipart("alternative")
        msg["Subject"]= f"[CloudSentinel] {severity} Anomaly — {instance_id}"
        msg["From"]   = ALERT_FROM
        msg["To"]     = ALERT_TO

        color   = SEVERITY_COLORS.get(severity, "#555")
        html_body = EMAIL_TEMPLATE.format(
            instance_id    = instance_id,
            detected_at    = detected_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            severity       = severity,
            ensemble_score = ensemble_score,
            if_score       = if_score,
            lstm_score     = lstm_score,
            color          = color,
        )
        msg.attach(MIMEText(html_body, "html"))
        return msg

    def _send_email(self, msg: MIMEMultipart) -> bool:
        """Send via SMTP with TLS. Returns True on success."""
        if not SMTP_USER or not SMTP_PASSWORD:
            logger.warning("SMTP credentials not configured — email skipped.")
            return False
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                recipients = [r.strip() for r in ALERT_TO.split(",") if r.strip()]
                server.sendmail(ALERT_FROM, recipients, msg.as_string())
            logger.info("Alert email sent to: %s", ALERT_TO)
            return True
        except Exception as exc:
            logger.error("Failed to send alert email: %s", exc)
            return False

    # ---- Main evaluate method ----------------------------------------

    def evaluate(
        self,
        instance_id:    str,
        ensemble_score: float,
        if_score:       float,
        lstm_score:     float,
        severity:       str,
        is_anomaly:     bool,
    ) -> Optional[int]:
        """
        Main entry point for the detection loop.

        1. Stores event in DB if anomaly.
        2. Sends email if not in cooldown.
        3. Returns event_id (or None if no anomaly).
        """
        if not is_anomaly:
            return None

        # Import here to avoid circular dependency at module level
        from database.db_utils import insert_anomaly_event, mark_alert_sent

        detected_at = datetime.now(timezone.utc)

        # 1. Persist to DB
        event_id = insert_anomaly_event(
            instance_id    = instance_id,
            if_score       = if_score,
            lstm_error     = lstm_score,
            ensemble_score = ensemble_score,
            severity       = severity,
            notes          = f"Auto-detected by CloudSentinel ensemble at {detected_at.isoformat()}",
        )
        logger.info(
            "Anomaly event stored | event_id=%d  instance=%s  "
            "severity=%s  score=%.4f",
            event_id, instance_id, severity, ensemble_score
        )

        # 2. Check cooldown
        if self._is_in_cooldown(instance_id):
            remaining = COOLDOWN_SECS - (
                datetime.now(timezone.utc) - self._cooldown_map[instance_id]
            ).total_seconds()
            logger.info(
                "Alert suppressed (cooldown %.0fs remaining) for %s",
                remaining, instance_id
            )
            return event_id

        # 3. Send email
        msg  = self._build_email(
            instance_id    = instance_id,
            detected_at    = detected_at,
            ensemble_score = ensemble_score,
            if_score       = if_score,
            lstm_score     = lstm_score,
            severity       = severity,
        )
        sent = self._send_email(msg)

        if sent:
            mark_alert_sent(event_id)
            self._record_alert(instance_id)

        return event_id
