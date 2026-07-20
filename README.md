# 🛡️ CloudSentinel

### Real-Time Anomaly Detection for Cloud Resource Monitoring Using Ensemble Machine Learning

> **MCA (AI/ML) Major Project** | 2025–2026

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

CloudSentinel is a real-time cloud infrastructure monitoring system that detects anomalies in CPU, memory, disk, and network metrics using a **weighted ensemble of Isolation Forest and LSTM Autoencoder** models — moving beyond brittle static-threshold alerts toward adaptive, ML-driven detection.

The system collects system metrics, learns normal behavior patterns, and flags deviations across four severity tiers (LOW / MEDIUM / HIGH), surfaced through a live Streamlit dashboard.

---

## 📁 Project Structure

```
cloudsentinel/
├── agents/
│   └── collector.py          # psutil-based metric collection agent
├── preprocessing/
│   └── preprocessor.py       # Normalization, windowing, feature extraction
├── models/
│   ├── isolation_forest.py   # Isolation Forest detector
│   ├── lstm_autoencoder.py   # LSTM Autoencoder detector
│   ├── ensemble.py           # Weighted ensemble combiner
│   └── saved/                # Trained model artifacts (gitignored)
├── alerting/
│   └── alert_engine.py       # Email alert dispatch with cooldown
├── dashboard/
│   └── app.py                # Streamlit real-time dashboard
├── database/
│   ├── schema.sql            # PostgreSQL schema
│   └── db_utils.py           # DB CRUD utilities
├── tests/
│   └── test_all.py           # Unit & integration tests
├── detection_loop.py         # Main real-time detection pipeline
├── train_all.py              # Model training script
├── evaluate_on_csv.py        # Offline evaluation against ground truth
├── requirements.txt
├── config.env.example
└── setup.sh
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- PostgreSQL 15+
- Windows / Linux / macOS (developed and tested locally on Windows)

### Quick Setup
```bash
git clone https://github.com/ShubhamOjha78/cloudsentinel.git
cd cloudsentinel
bash setup.sh
```

### Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp config.env.example config.env
# Fill in your DB credentials and SMTP details in config.env

# 3. Initialize database
psql -U postgres -f database/schema.sql
```

---

## 🚀 Usage

```bash
# 1. Start metric collection
python agents/collector.py --instance-id local-01 --interval 30

# 2. Train models (after collecting some data)
python train_all.py --instance-id local-01

# 3. Start the real-time detection loop
python detection_loop.py --instance-id local-01 --interval 30

# 4. Launch the dashboard
streamlit run dashboard/app.py --server.port 8501
# Open: http://localhost:8501
```

Run tests:
```bash
python -m pytest tests/ -v
```

---

## 🧠 ML Architecture

| Component | Algorithm | Purpose |
|-----------|-----------|---------|
| Detector 1 | Isolation Forest | Structural / point anomaly detection |
| Detector 2 | LSTM Autoencoder | Temporal / sequence anomaly detection |
| Ensemble | Weighted Average (IF × 0.40 + LSTM × 0.60) | Combined robust decision |

**Ensemble Score Formula**
```
ensemble_score = 0.40 × IF_score + 0.60 × LSTM_reconstruction_error
```

**Key constants:** `WINDOW_SIZE = 60`, `TRAIN_RATIO = 0.60`, anomaly threshold = `0.65`

**Features monitored:** `cpu_percent`, `mem_percent`, `disk_read_mb`, `disk_write_mb`, `net_recv_kb`, `net_sent_kb`

---

## 📊 Evaluation Results

Evaluated on `cloud_metrics.csv` (855 rows, 342 test samples, Z-score-derived ground truth):

| Model | Precision | Recall | F1-Score | FPR |
|-------|-----------|--------|----------|-----|
| Z-Score Baseline | 100.00% | 19.44% | 32.56% | 0.00% |
| Isolation Forest | 35.56% | 88.89% | 50.79% | 23.48% |
| LSTM Autoencoder | 35.00% | 58.33% | 43.75% | 15.79% |
| **CloudSentinel Ensemble** | **39.62%** | **58.33%** | **47.19%** | **12.96%** |

**Why the ensemble, despite a slightly lower F1 than standalone Isolation Forest?**
The ensemble achieves the **lowest false positive rate (12.96%)** among all trained models — nearly half of Isolation Forest's 23.48%. In a production monitoring system, false alerts are costly (alert fatigue), making the ensemble the more practical choice even at a modest precision/recall trade-off.

> **Note:** This was developed and evaluated locally; AWS EC2 deployment is a planned future extension, not yet implemented.

---

## 🔒 Security

- Credentials stored in `config.env` (never hardcoded), excluded via `.gitignore`
- Trained model artifacts (`.pkl`, `.h5`) excluded from version control
- PostgreSQL used for structured metric and alert storage

---

## 📚 Technologies Used

Python · scikit-learn · TensorFlow/Keras · pandas · NumPy · PostgreSQL · psycopg2 · Streamlit · psutil · Git

---

## 🔭 Future Work

- Deploy on AWS EC2 for live cloud-instance monitoring
- Expand alerting channels (Slack/SMS)
- Hyperparameter tuning for ensemble weights

---

## 👤 Author

**Shubham** | Enrollment No. O24MCA111851
MCA (Artificial Intelligence & Machine Learning)
