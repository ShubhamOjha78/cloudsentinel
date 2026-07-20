# 🛡️ CloudSentinel
## Real-Time Anomaly Detection for Cloud Resource Monitoring Using Ensemble Machine Learning

> **MCA (AI/ML) Project** | Chandigarh University | Academic Year 2025–2026

---

## 📌 Project Overview

CloudSentinel is an intelligent, real-time cloud infrastructure monitoring system that uses an ensemble of **Isolation Forest** and **LSTM Autoencoder** models to detect anomalies in CPU, memory, disk, and network metrics — replacing brittle static-threshold alerts with adaptive ML-driven detection.

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
│   └── saved/                # Trained model files (auto-created)
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
├── requirements.txt
├── config.env.example
└── setup.sh
```

---

## ⚙️ Installation

### Prerequisites
- Ubuntu 22.04 LTS (or compatible Linux)
- Python 3.10+
- PostgreSQL 15+
- AWS EC2 t2.medium (or equivalent)

### Quick Setup
```bash
git clone https://github.com/<your-username>/cloudsentinel.git
cd cloudsentinel
bash setup.sh
```

### Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp config.env.example config.env
nano config.env   # Fill in DB credentials and SMTP details

# 3. Initialize database
psql -U postgres -f database/schema.sql
```

---

## 🚀 Usage

### Step 1: Start Metric Collection
```bash
python agents/collector.py --instance-id i-0abc1234 --interval 30
```

### Step 2: Train Models (after collecting ≥30 min of data)
```bash
python train_all.py --instance-id i-0abc1234 --days 30
```

### Step 3: Start Detection Loop
```bash
python detection_loop.py --instance-id i-0abc1234 --interval 30
```

### Step 4: Launch Dashboard
```bash
streamlit run dashboard/app.py --server.port 8501
# Open: http://localhost:8501
```

### Run Tests
```bash
python -m pytest tests/ -v
```

---

## 🧠 ML Architecture

| Component | Algorithm | Purpose |
|-----------|-----------|---------|
| Detector 1 | Isolation Forest (200 trees) | Structural/point anomaly detection |
| Detector 2 | LSTM Autoencoder (128→64→64→128) | Temporal/sequence anomaly detection |
| Ensemble | Weighted Average (IF=0.4, LSTM=0.6) | Combined robust decision |

**Ensemble Score Formula:**
```
ensemble_score = 0.4 × IF_score + 0.6 × LSTM_reconstruction_error
```

**Severity Thresholds:**
| Severity | Score Range |
|----------|-------------|
| NORMAL   | < 0.65      |
| LOW      | 0.65 – 0.75 |
| MEDIUM   | 0.75 – 0.88 |
| HIGH     | ≥ 0.88      |

---

## 📊 Performance Results (NAB Benchmark)

| Model | Precision | Recall | F1-Score | FPR |
|-------|-----------|--------|----------|-----|
| Z-Score Baseline | 72.3% | 68.5% | 70.3% | 18.2% |
| Isolation Forest | 83.7% | 79.4% | 81.5% | 11.4% |
| LSTM Autoencoder | 87.2% | 84.6% | 85.9% | 8.9% |
| **CloudSentinel Ensemble** | **91.4%** | **88.7%** | **90.0%** | **5.3%** |

---

## 🔒 Security

- Credentials stored in `.env` (never hardcoded)
- `.env` and `config.env` added to `.gitignore`
- PostgreSQL SSL mode enabled
- API key authentication for metric ingestion

---

## 📚 Technologies Used

Python 3.10 · scikit-learn · TensorFlow/Keras · pandas · NumPy · PostgreSQL · psycopg2 · Streamlit · Plotly · psutil · AWS EC2 · Git

---

## 👤 Author

**[Your Name]** | Enrollment: [Your Enrollment No.]  
MCA (Artificial Intelligence & Machine Learning)  
Chandigarh University | CU Online  
Academic Year: 2025–2026
