-- ============================================================
-- CloudSentinel: Database Schema
-- Project: Real-Time Anomaly Detection for Cloud Monitoring
-- Author : [Your Name] | MCA (AI/ML) | Chandigarh University
-- ============================================================

-- Drop existing tables (for fresh setup)
DROP TABLE IF EXISTS anomaly_events CASCADE;
DROP TABLE IF EXISTS cloud_metrics CASCADE;

-- -------------------------------------------------------
-- Table 1: cloud_metrics
-- Stores raw time-series metrics from cloud instances
-- -------------------------------------------------------
CREATE TABLE cloud_metrics (
    metric_id       SERIAL          PRIMARY KEY,
    instance_id     VARCHAR(50)     NOT NULL,
    timestamp       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    cpu_percent     FLOAT           CHECK (cpu_percent BETWEEN 0 AND 100),
    mem_percent     FLOAT           CHECK (mem_percent BETWEEN 0 AND 100),
    disk_read_mb    FLOAT           NOT NULL DEFAULT 0,
    disk_write_mb   FLOAT           NOT NULL DEFAULT 0,
    net_recv_kb     FLOAT           NOT NULL DEFAULT 0,
    net_sent_kb     FLOAT           NOT NULL DEFAULT 0
);

-- Index for fast time-range queries per instance
CREATE INDEX idx_metrics_instance_time
    ON cloud_metrics (instance_id, timestamp DESC);

-- -------------------------------------------------------
-- Table 2: anomaly_events
-- Stores detected anomaly records with model scores
-- -------------------------------------------------------
CREATE TABLE anomaly_events (
    event_id        SERIAL          PRIMARY KEY,
    instance_id     VARCHAR(50)     NOT NULL,
    detected_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    if_score        FLOAT           NOT NULL,
    lstm_error      FLOAT           NOT NULL,
    ensemble_score  FLOAT           NOT NULL,
    severity        VARCHAR(10)     CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH')),
    alert_sent      BOOLEAN         DEFAULT FALSE,
    notes           TEXT
);

-- Index for querying recent anomalies per instance
CREATE INDEX idx_anomaly_instance_time
    ON anomaly_events (instance_id, detected_at DESC);

-- -------------------------------------------------------
-- Table 3: model_registry
-- Tracks trained model versions
-- -------------------------------------------------------
CREATE TABLE model_registry (
    model_id        SERIAL          PRIMARY KEY,
    model_type      VARCHAR(30)     NOT NULL,  -- 'isolation_forest' or 'lstm'
    instance_id     VARCHAR(50)     NOT NULL,
    trained_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    model_path      TEXT            NOT NULL,
    f1_score        FLOAT,
    precision_score FLOAT,
    recall_score    FLOAT,
    is_active       BOOLEAN         DEFAULT TRUE
);

COMMENT ON TABLE cloud_metrics    IS 'Raw time-series metrics from cloud instances';
COMMENT ON TABLE anomaly_events   IS 'Detected anomaly events with ML scores';
COMMENT ON TABLE model_registry   IS 'Registry of trained ML model versions';
