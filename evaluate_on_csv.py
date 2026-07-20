"""
CloudSentinel – evaluate_on_csv.py
================================================
NAB dataset ya custom CSV pe model evaluation.

Usage:
    # NAB dataset (recommended)
    python evaluate_on_csv.py --csv ec2_cpu_utilization_5f5533.csv --mode nab

    # Custom CSV
    python evaluate_on_csv.py --csv system_performance_metrics.csv --mode zscore
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

FEATURE_COLS = ["cpu_percent", "mem_percent", "disk_read_mb",
                "disk_write_mb", "net_recv_kb", "net_sent_kb"]
WINDOW_SIZE  = 60
TRAIN_RATIO  = 0.60
IF_WEIGHT    = 0.40
LSTM_WEIGHT  = 0.60
THRESHOLD    = 0.65
Z_THRESHOLD  = 2.5

# NAB known anomaly windows
NAB_ANOMALY_WINDOWS = {
    "ec2_cpu_utilization_5f5533": [
        ("2014-04-10 07:15:00", "2014-04-11 16:45:00")
    ],
    "ec2_cpu_utilization_ac20cd": [
        ("2014-04-13 09:00:00", "2014-04-14 18:30:00")
    ],
    "ec2_cpu_utilization_c6585a": [
        ("2014-04-17 09:30:00", "2014-04-18 19:00:00")
    ],
    "ec2_disk_write_bytes_1ef3de": [
        ("2014-03-14 09:00:00", "2014-03-15 18:30:00")
    ],
    "ec2_network_in_5abac7": [
        ("2014-04-10 07:00:00", "2014-04-11 16:30:00")
    ],
}


def load_nab_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "value" in df.columns:
        df["cpu_percent"] = df["value"]
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()
    logger.info("NAB CSV loaded | rows=%d", len(df))
    return df


def load_custom_csv(path):
    col_map = {
        "cpu_usage": "cpu_percent", "cpu_utilization": "cpu_percent",
        "memory_usage": "mem_percent", "memory_utilization": "mem_percent",
        "disk_usage": "disk_read_mb", "disk_read": "disk_read_mb",
        "disk_write": "disk_write_mb",
        "network_in": "net_recv_kb", "network_out": "net_sent_kb",
    }
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=col_map)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()
    logger.info("Custom CSV loaded | rows=%d", len(df))
    return df


def generate_nab_labels(df, csv_path):
    labels   = np.zeros(len(df), dtype=int)
    csv_name = Path(csv_path).stem
    windows  = None
    for key in NAB_ANOMALY_WINDOWS:
        if key in csv_name:
            windows = NAB_ANOMALY_WINDOWS[key]
            break
    if windows is None:
        logger.warning("NAB windows not found for '%s' — using Z-score", csv_name)
        return generate_zscore_labels(df)
    for start_str, end_str in windows:
        start = pd.Timestamp(start_str)
        end   = pd.Timestamp(end_str)
        mask  = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        labels[mask] = 1
        logger.info("Anomaly window: %s to %s | %d rows", start_str, end_str, mask.sum())
    if labels.sum() == 0:
        logger.warning("Window outside data range! Data: %s to %s — using Z-score",
            df["timestamp"].min().strftime("%Y-%m-%d"),
            df["timestamp"].max().strftime("%Y-%m-%d"))
        return generate_zscore_labels(df)
    logger.info("NAB labels | anomalies=%d (%.1f%%)", labels.sum(), labels.mean()*100)
    return labels


def generate_zscore_labels(df, z_thresh=Z_THRESHOLD):
    labels = np.zeros(len(df), dtype=int)
    valid_cols = [c for c in FEATURE_COLS if df[c].std() > 0]
    for vcol in valid_cols:
        z = np.abs((df[vcol] - df[vcol].mean()) / df[vcol].std())
        labels[z > z_thresh] = 1
    logger.info("Z-score labels | anomalies=%d (%.1f%%)", labels.sum(), labels.mean()*100)
    return labels


def split_data(df, labels):
    n       = int(len(df) * TRAIN_RATIO)
    df_tr   = df.iloc[:n].reset_index(drop=True)
    df_te   = df.iloc[n:].reset_index(drop=True)
    lbl_te  = labels[n:]
    logger.info("Split | train=%d test=%d | test_anomalies=%d", len(df_tr), len(df_te), lbl_te.sum())
    return df_tr, df_te, lbl_te


def normalize(df_train, df_test):
    scaler = MinMaxScaler()
    scaler.fit(df_train[FEATURE_COLS])
    tr, te = df_train.copy(), df_test.copy()
    tr[FEATURE_COLS] = scaler.transform(df_train[FEATURE_COLS])
    te[FEATURE_COLS] = scaler.transform(df_test[FEATURE_COLS])
    return tr, te


def make_windows(df_norm):
    data = df_norm[FEATURE_COLS].values.astype("float32")
    wins, flts, idxs = [], [], []
    for i in range(WINDOW_SIZE, len(data) + 1):
        wins.append(data[i - WINDOW_SIZE: i])
        flts.append(data[i - 1])
        idxs.append(i - 1)
    return np.array(wins), np.array(flts), idxs


def run_isolation_forest(train_flats, test_flats):
    logger.info("Training Isolation Forest...")
    clf = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=42, n_jobs=-1)
    clf.fit(train_flats)
    raw  = clf.decision_function(train_flats)
    smin, smax = raw.min(), raw.max()
    def norm(r):
        return np.clip((-r - (-smax)) / max((-smin)-(-smax), 1e-9), 0, 1)
    scores = norm(clf.decision_function(test_flats))
    logger.info("IF done | mean=%.4f", scores.mean())
    return scores


def run_lstm(train_wins, test_wins):
    logger.info("Training LSTM (2-3 min)...")
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (Input, LSTM, Dense,
                                             RepeatVector, TimeDistributed)
        from tensorflow.keras.callbacks import EarlyStopping

        inp = Input(shape=(WINDOW_SIZE, len(FEATURE_COLS)))
        x   = LSTM(64, return_sequences=True)(inp)
        x   = LSTM(32, return_sequences=False)(x)
        x   = RepeatVector(WINDOW_SIZE)(x)
        x   = LSTM(32, return_sequences=True)(x)
        x   = LSTM(64, return_sequences=True)(x)
        out = TimeDistributed(Dense(len(FEATURE_COLS)))(x)
        mdl = Model(inp, out)
        mdl.compile(optimizer="adam", loss="mse")
        mdl.fit(train_wins, train_wins, epochs=30, batch_size=32,
                validation_split=0.1,
                callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
                verbose=1)
        pred   = mdl.predict(test_wins, verbose=0, batch_size=32)
        errors = np.mean(np.square(test_wins - pred), axis=(1, 2))
        scores = np.clip((errors - errors.min()) / max(errors.max()-errors.min(), 1e-9), 0, 1)
        logger.info("LSTM done | mean_error=%.4f", errors.mean())
        return scores
    except Exception as e:
        logger.warning("LSTM failed: %s — using zeros", e)
        return np.zeros(len(test_wins))


def evaluate(y_true, y_pred_scores, model_name):
    y_pred  = (y_pred_scores >= THRESHOLD).astype(int)
    min_len = min(len(y_true), len(y_pred))
    y_true  = y_true[-min_len:]
    y_pred  = y_pred[-min_len:]
    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape==(2,2) else (0,0,0,0)
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0
    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  Precision : {p*100:.2f}%")
    print(f"  Recall    : {r*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  FP Rate   : {fpr*100:.2f}%")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(classification_report(y_true, y_pred,
                                target_names=["Normal","Anomaly"], zero_division=0))
    return {"model": model_name, "precision": p, "recall": r, "f1": f1, "fpr": fpr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    required=True)
    parser.add_argument("--mode",   default="nab", choices=["nab","zscore"])
    parser.add_argument("--output", default="evaluation_report.txt")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  CloudSentinel — Model Evaluation Report")
    print("  MCA (AI/ML) | Chandigarh University")
    print("="*55)
    print(f"  Dataset : {args.csv}")
    print(f"  Mode    : {args.mode}")
    print("="*55)

    df     = load_nab_csv(args.csv) if args.mode=="nab" else load_custom_csv(args.csv)
    labels = generate_nab_labels(df, args.csv) if args.mode=="nab" else generate_zscore_labels(df)

    df_tr, df_te, lbl_te = split_data(df, labels)
    df_trn, df_ten       = normalize(df_tr, df_te)

    tr_w, tr_f, _       = make_windows(df_trn)
    te_w, te_f, te_idx  = make_windows(df_ten)
    lbl_win             = lbl_te[te_idx]

    # Z-score baseline
    base_full = np.zeros(len(df_te))
    valid_cols = [c for c in FEATURE_COLS if df_tr[c].std() > 0]
    for vcol in valid_cols:
        z         = np.abs((df_te[vcol].values - df_tr[vcol].mean()) / df_tr[vcol].std())
        base_full = np.maximum(base_full, z / 10.0)
    base = base_full[te_idx]   # align to window indices
    r0 = evaluate(lbl_win, base, "Z-Score Baseline")

    # IF
    if_s = run_isolation_forest(tr_f, te_f)
    r1   = evaluate(lbl_win, if_s, "Isolation Forest")

    # LSTM
    lstm_s = run_lstm(tr_w, te_w)
    r2     = evaluate(lbl_win, lstm_s, "LSTM Autoencoder")

    # Ensemble
    ens  = (IF_WEIGHT * if_s) + (LSTM_WEIGHT * lstm_s)
    r3   = evaluate(lbl_win, ens, "CloudSentinel Ensemble (IF x0.4 + LSTM x0.6)")

    # Summary
    results = [r0, r1, r2, r3]
    print("\n" + "="*58)
    print("  FINAL COMPARISON TABLE")
    print("="*58)
    print(f"{'Model':<40} {'P':>5} {'R':>5} {'F1':>5} {'FPR':>5}")
    print("-"*58)
    for r in results:
        print(f"{r['model']:<40} {r['precision']*100:>4.1f}% "
              f"{r['recall']*100:>4.1f}% {r['f1']*100:>4.1f}% "
              f"{r['fpr']*100:>4.1f}%")
    print("="*58)

    lines = [
        "CloudSentinel — Model Evaluation Report",
        "MCA (AI/ML) | Chandigarh University",
        f"Dataset : {args.csv}",
        f"Mode    : {args.mode}",
        f"Rows    : {len(df)} | Train: {len(df_tr)} | Test: {len(df_te)}",
        f"Anomalies in test: {lbl_win.sum()} ({lbl_win.mean()*100:.1f}%)",
        "",
        f"{'Model':<40} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FPR':>8}",
        "-"*72,
    ]
    for r in results:
        lines.append(f"{r['model']:<40} {r['precision']*100:>9.2f}% "
                     f"{r['recall']*100:>7.2f}% {r['f1']*100:>7.2f}% "
                     f"{r['fpr']*100:>7.2f}%")

    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    logger.info("Report saved → %s", args.output)
    print(f"\n✅ Report saved: {args.output}")


if __name__ == "__main__":
    main()
