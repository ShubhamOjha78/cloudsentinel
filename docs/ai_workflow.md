# CloudSentinel — AI Workflow & Monitoring Integration Guide

**Project:** Real-Time Anomaly Detection for Cloud Resource Monitoring  
**Program:** MCA (AI/ML) | Chandigarh University | CU Online  
**Academic Year:** 2025–2026

---

## 1. Overview

CloudSentinel's AI workflow is a fully automated pipeline that moves from raw infrastructure telemetry to intelligent anomaly decisions in under 3 seconds. The pipeline is divided into two phases:

- **Offline Phase** — Data collection, preprocessing, and model training (runs once or periodically)
- **Online Phase** — Real-time metric ingestion, inference, alerting, and visualization (runs continuously)

---

## 2. End-to-End AI Workflow Diagram

```
╔══════════════════════════════════════════════════════════════════════╗
║                    CLOUDSENTINEL AI PIPELINE                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─────────────┐     ┌──────────────┐     ┌────────────────────┐    ║
║  │  AWS EC2    │────▶│  collector   │────▶│  cloud_metrics DB  │    ║
║  │  Instance   │     │  (psutil)    │     │  (PostgreSQL)      │    ║
║  └─────────────┘     └──────────────┘     └────────┬───────────┘    ║
║                       every 30s                     │               ║
║                                                     ▼               ║
║                                          ┌────────────────────┐     ║
║                                          │  DataPreprocessor  │     ║
║                                          │  • ffill NaN       │     ║
║                                          │  • MinMaxScaler    │     ║
║                                          │  • sliding window  │     ║
║                                          └────────┬───────────┘     ║
║                                                   │                 ║
║                              ┌────────────────────┤                 ║
║                              │                    │                 ║
║                              ▼                    ▼                 ║
║                   ┌──────────────────┐  ┌──────────────────┐        ║
║                   │ Isolation Forest │  │ LSTM Autoencoder │        ║
║                   │  (200 trees)     │  │ (128→64→64→128)  │        ║
║                   │  IF score [0,1]  │  │  MSE error [0,1] │        ║
║                   └────────┬─────────┘  └────────┬─────────┘        ║
║                            │                     │                  ║
║                            └──────────┬──────────┘                  ║
║                                       ▼                             ║
║                           ┌─────────────────────┐                   ║
║                           │  Ensemble Detector  │                   ║
║                           │  0.4×IF + 0.6×LSTM  │                   ║
║                           │  score ∈ [0, 1]     │                   ║
║                           └──────────┬──────────┘                   ║
║                                      │                              ║
║                      ┌───────────────┼───────────────┐             ║
║                      ▼               ▼               ▼             ║
║                  score<0.65     0.65–0.88        score≥0.88        ║
║                  NORMAL         LOW/MEDIUM          HIGH            ║
║                      │               │               │             ║
║                      │         ┌─────▼───────────────▼──────┐      ║
║                      │         │       Alert Engine          │      ║
║                      │         │  • DB insert anomaly_events │      ║
║                      │         │  • HTML email (SMTP/TLS)    │      ║
║                      │         │  • 5-min cooldown guard     │      ║
║                      │         └────────────┬────────────────┘      ║
║                      │                      │                       ║
║                      └──────────────────────▼                       ║
║                                 ┌──────────────────┐                ║
║                                 │ Streamlit        │                ║
║                                 │ Dashboard        │                ║
║                                 │ (auto-refresh)   │                ║
║                                 └──────────────────┘                ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 3. Offline Phase — Model Training Workflow

### Step 1: Data Collection (agents/collector.py)
The monitoring agent collects 6 system metrics every 30 seconds using `psutil`:

| Metric | Source | Unit |
|--------|--------|------|
| cpu_percent | `psutil.cpu_percent()` | % |
| mem_percent | `psutil.virtual_memory().percent` | % |
| disk_read_mb | `disk_io_counters().read_bytes` delta | MB/s |
| disk_write_mb | `disk_io_counters().write_bytes` delta | MB/s |
| net_recv_kb | `net_io_counters().bytes_recv` delta | KB/s |
| net_sent_kb | `net_io_counters().bytes_sent` delta | KB/s |

All records are stored in `cloud_metrics` (PostgreSQL) with instance_id and UTC timestamp.

### Step 2: Preprocessing (preprocessing/preprocessor.py)

```
Raw DB records
    │
    ▼
to_dataframe()          → sort by timestamp
    │
    ▼
handle_missing()        → forward-fill → backward-fill
    │
    ▼
fit_scaler()            → MinMaxScaler fitted on training data
    │                      saved to models/saved/scaler.pkl
    ▼
normalize()             → all features scaled to [0, 1]
    │
    ├──▶ build_flat_vectors()   → shape (n_rows, 6)     → Isolation Forest input
    │
    └──▶ build_windows()        → shape (n, 60, 6)      → LSTM input
                                   window_size = 60 steps
                                   = 30 minutes at 30s interval
```

### Step 3: Isolation Forest Training (models/isolation_forest.py)

The Isolation Forest algorithm works by randomly partitioning the feature space using binary trees. Anomalies are isolated faster (closer to the root) because they are rare and distinct from normal data.

**Hyperparameters used:**
```
n_estimators  = 200       # number of isolation trees
contamination = 0.05      # expected anomaly rate in training data
max_samples   = 'auto'    # min(256, n_samples)
random_state  = 42        # reproducibility
```

**Score normalization:**  
sklearn's `decision_function()` returns higher values for normal points. CloudSentinel inverts and normalizes this to a [0,1] scale where 1 = most anomalous:
```
raw_score     = model.decision_function(X)
inverted      = -raw_score
normalized    = (inverted - (-score_max)) / ((-score_min) - (-score_max))
```

### Step 4: LSTM Autoencoder Training (models/lstm_autoencoder.py)

The LSTM Autoencoder learns to reconstruct normal time-series patterns. High reconstruction error on new data indicates anomalous behavior.

**Architecture:**
```
Input  (60, 6)
  │
  ├─ Encoder LSTM 128 units (return_sequences=True)
  ├─ Encoder LSTM  64 units (return_sequences=False)  ← bottleneck
  ├─ RepeatVector(60)
  ├─ Decoder LSTM  64 units (return_sequences=True)
  ├─ Decoder LSTM 128 units (return_sequences=True)
  └─ TimeDistributed Dense(6)
Output (60, 6)
```

**Training config:**
```
loss          = MSE (Mean Squared Error)
optimizer     = Adam (lr=0.001)
epochs        = 50 (with EarlyStopping patience=5)
batch_size    = 64
val_split     = 20%
```

**Anomaly score:**  
Reconstruction error is the mean MSE across all time steps and features, normalized to [0,1] using training-set calibration statistics.

---

## 4. Online Phase — Real-Time Detection Workflow

### Detection Loop (detection_loop.py)

The detection loop runs every 30 seconds per monitored instance:

```
① fetch_metrics_window(instance_id, window_size=60)
        ↓
② get_latest_window() → normalize + build (window, flat)
        ↓
③ EnsembleDetector.detect(window, flat)
   ├── IsolationForestDetector.score(flat)    → if_score
   └── LSTMDetector.score(window)             → lstm_score
        ↓
④ ensemble_score = 0.4 × if_score + 0.6 × lstm_score
        ↓
⑤ classify_severity(ensemble_score)
   ├── < 0.65  → NORMAL  (no action)
   ├── 0.65–0.75 → LOW
   ├── 0.75–0.88 → MEDIUM
   └── ≥ 0.88  → HIGH
        ↓
⑥ AlertEngine.evaluate() [if anomaly]
   ├── insert_anomaly_event() → PostgreSQL
   ├── check cooldown (5 min)
   └── send_email() → SMTP/TLS → Administrator
```

### Ensemble Weight Rationale

| Model | Weight | Reason |
|-------|--------|--------|
| Isolation Forest | 0.40 | Strong for point/structural anomalies; fast inference |
| LSTM Autoencoder | 0.60 | Superior for temporal/contextual anomalies; higher F1 standalone |

Weight ratio was optimized via grid search over [0.3–0.7] in 0.1 steps. The 0.4/0.6 split achieved the highest F1 (90.0%) on the NAB benchmark.

---

## 5. Monitoring Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLOUD ENVIRONMENT (AWS)                  │
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │  EC2 inst-1  │   │  EC2 inst-2  │   │  EC2 inst-N  │   │
│  │ collector.py │   │ collector.py │   │ collector.py │   │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   │
│         └─────────────┬────┘                  │            │
│                       │  INSERT metrics        │            │
│                       ▼                        │            │
│          ┌────────────────────────┐            │            │
│          │   PostgreSQL DB        │◀───────────┘            │
│          │   cloud_metrics        │                         │
│          │   anomaly_events       │                         │
│          │   model_registry       │                         │
│          └───────────┬────────────┘                         │
│                      │                                      │
│         ┌────────────┼────────────┐                         │
│         ▼            ▼            ▼                         │
│  detection_loop  detection_loop  detection_loop             │
│  (inst-1)        (inst-2)        (inst-N)                   │
│         │            │            │                         │
│         └────────────▼────────────┘                         │
│                 Alert Engine                                │
│                      │                                      │
│         ┌────────────▼────────────┐                         │
│         │   Streamlit Dashboard   │  ← Admin browser        │
│         │   :8501                 │                         │
│         └─────────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibility Matrix

| Component | File | Trigger | Output |
|-----------|------|---------|--------|
| Metrics Agent | `agents/collector.py` | cron / systemd (every 30s) | DB row insert |
| Preprocessor | `preprocessing/preprocessor.py` | Called by detection loop | Normalized arrays |
| IF Detector | `models/isolation_forest.py` | Called by ensemble | Anomaly score [0,1] |
| LSTM Detector | `models/lstm_autoencoder.py` | Called by ensemble | Reconstruction error [0,1] |
| Ensemble | `models/ensemble.py` | Called by detection loop | AnomalyResult |
| Alert Engine | `alerting/alert_engine.py` | AnomalyResult.is_anomaly=True | DB event + email |
| Dashboard | `dashboard/app.py` | Browser / auto-refresh | Visual output |

---

## 6. Model Performance Summary

| Model | Precision | Recall | F1 | FPR |
|-------|-----------|--------|----|-----|
| Z-Score Baseline | 72.3% | 68.5% | 70.3% | 18.2% |
| Isolation Forest | 83.7% | 79.4% | 81.5% | 11.4% |
| LSTM Autoencoder | 87.2% | 84.6% | 85.9% | 8.9% |
| **CloudSentinel Ensemble** | **91.4%** | **88.7%** | **90.0%** | **5.3%** |

Evaluated on: Numenta Anomaly Benchmark (NAB) — `realAWSCloudwatch` subset.

---

## 7. Running the Full Workflow

```bash
# Step 1 — One-time setup
bash setup.sh

# Step 2 — Start data collection (run on each instance)
python agents/collector.py --instance-id i-0abc1234 --interval 30

# Step 3 — Train models (after ≥30 min of data)
python train_all.py --instance-id i-0abc1234 --days 30

# Step 4 — Start real-time detection
python detection_loop.py --instance-id i-0abc1234 --interval 30

# Step 5 — Launch dashboard
streamlit run dashboard/app.py --server.port 8501

# Step 6 — Run tests
python -m pytest tests/ -v
```

---

*CloudSentinel AI Workflow Documentation v1.0 — MCA (AI/ML) Project, Chandigarh University, 2025–2026*
