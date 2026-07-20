"""
CloudSentinel – evaluate_from_db.py
=====================================================
Direct database se data lekar model evaluation karo.
CSV ki zaroorat nahi — PostgreSQL se seedha data aata hai.

Usage:
    # Saare instances ka evaluation
    python evaluate_from_db.py

    # Specific instance
    python evaluate_from_db.py --instance my-windows-pc

    # Custom date range
    python evaluate_from_db.py --start 2026-05-09 --end 2026-05-24

    # Report file save karo
    python evaluate_from_db.py --output my_eval_report.txt

MCA (AI/ML) | Chandigarh University | CloudSentinel Project
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ── Apne project ka database module load karo ───────────────────
# Ye script cloudsentinel folder ke andar rakho
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from database.db_utils import (
        get_connection,
        get_distinct_instances,
        DB_CONFIG
    )
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants (same as evaluate_on_csv.py) ──────────────────────
FEATURE_COLS = [
    "cpu_percent", "mem_percent",
    "disk_read_mb", "disk_write_mb",
    "net_recv_kb", "net_sent_kb"
]
WINDOW_SIZE = 60
TRAIN_RATIO = 0.60
IF_WEIGHT   = 0.40
LSTM_WEIGHT = 0.60
THRESHOLD   = 0.65
Z_THRESHOLD = 2.5


# ════════════════════════════════════════════════════════════════
# STEP 1 — Database se data fetch karo
# ════════════════════════════════════════════════════════════════

def fetch_from_db(instance_id=None, start_dt=None, end_dt=None):
    """
    cloud_metrics table se seedha pandas DataFrame banao.
    instance_id aur date range optional hain.
    """
    if not DB_AVAILABLE:
        raise ImportError(
            "database/db_utils.py nahi mila.\n"
            "Ye script apne cloudsentinel project folder mein rakho."
        )

    # Dynamic SQL query build karo
    conditions = []
    params     = []

    if instance_id:
        conditions.append("instance_id = %s")
        params.append(instance_id)

    if start_dt:
        conditions.append("timestamp >= %s")
        params.append(start_dt)

    if end_dt:
        conditions.append("timestamp <= %s")
        params.append(end_dt)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT
            metric_id,
            instance_id,
            timestamp,
            cpu_percent,
            mem_percent,
            disk_read_mb,
            disk_write_mb,
            net_recv_kb,
            net_sent_kb
        FROM cloud_metrics
        {where}
        ORDER BY timestamp ASC
    """

    logger.info("Database se data fetch ho raha hai...")
    logger.info("Config: host=%s db=%s user=%s",
                DB_CONFIG['host'], DB_CONFIG['dbname'], DB_CONFIG['user'])

    import psycopg2.extras
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params if params else None)
            rows = cur.fetchall()

    if not rows:
        raise ValueError(
            "Database mein koi data nahi mila!\n"
            "Check karo:\n"
            "  1. collector.py chal raha tha?\n"
            "  2. Instance ID sahi hai?\n"
            "  3. Date range mein data hai?"
        )

    df = pd.DataFrame([dict(r) for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Missing values fill karo
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()

    logger.info("✓ Data loaded | rows=%d | instances=%s",
                len(df), df["instance_id"].unique().tolist())
    logger.info("  Time range: %s  →  %s",
                df["timestamp"].min().strftime("%Y-%m-%d %H:%M"),
                df["timestamp"].max().strftime("%Y-%m-%d %H:%M"))

    return df


def show_db_summary(df):
    """Data ka brief summary print karo before evaluation."""
    print("\n" + "─"*55)
    print("  📊 DATABASE DATA SUMMARY")
    print("─"*55)
    print(f"  Total rows        : {len(df):,}")
    print(f"  Instances         : {', '.join(df['instance_id'].unique())}")
    print(f"  From              : {df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  To                : {df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration          : {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    print("─"*55)
    print("  METRIC STATISTICS (mean ± std):")
    for col in FEATURE_COLS:
        print(f"  {col:<18}: {df[col].mean():>7.2f} ± {df[col].std():.2f}")
    print("─"*55)


# ════════════════════════════════════════════════════════════════
# STEP 2 — Anomaly labels banao (Z-score method)
# ════════════════════════════════════════════════════════════════

def generate_labels(df, z_thresh=Z_THRESHOLD):
    """
    Z-score se ground truth labels banao.
    Koi bhi metric jo z_thresh se zyada spike kare = anomaly.
    """
    labels = np.zeros(len(df), dtype=int)
    valid_cols = [c for c in FEATURE_COLS if df[c].std() > 0]

    for col in valid_cols:
        z = np.abs((df[col] - df[col].mean()) / df[col].std())
        labels[z > z_thresh] = 1

    logger.info("Z-score labels | anomalies=%d / %d (%.1f%%)",
                labels.sum(), len(labels), labels.mean() * 100)
    return labels


# ════════════════════════════════════════════════════════════════
# STEP 3 — Data split + normalize
# ════════════════════════════════════════════════════════════════

def split_and_normalize(df, labels):
    n      = int(len(df) * TRAIN_RATIO)
    df_tr  = df.iloc[:n].reset_index(drop=True)
    df_te  = df.iloc[n:].reset_index(drop=True)
    lbl_te = labels[n:]

    scaler = MinMaxScaler()
    scaler.fit(df_tr[FEATURE_COLS])

    df_trn, df_ten = df_tr.copy(), df_te.copy()
    df_trn[FEATURE_COLS] = scaler.transform(df_tr[FEATURE_COLS])
    df_ten[FEATURE_COLS] = scaler.transform(df_te[FEATURE_COLS])

    logger.info("Split | train=%d test=%d | test_anomalies=%d",
                len(df_tr), len(df_te), lbl_te.sum())
    return df_tr, df_te, df_trn, df_ten, lbl_te


# ════════════════════════════════════════════════════════════════
# STEP 4 — Sliding windows banao
# ════════════════════════════════════════════════════════════════

def make_windows(df_norm):
    data = df_norm[FEATURE_COLS].values.astype("float32")
    wins, flats, idxs = [], [], []
    for i in range(WINDOW_SIZE, len(data) + 1):
        wins.append(data[i - WINDOW_SIZE: i])
        flats.append(data[i - 1])
        idxs.append(i - 1)
    return np.array(wins), np.array(flats), idxs


# ════════════════════════════════════════════════════════════════
# STEP 5 — Models chalao
# ════════════════════════════════════════════════════════════════

def run_isolation_forest(train_flats, test_flats):
    logger.info("Isolation Forest training...")
    clf = IsolationForest(
        n_estimators=200, contamination=0.05,
        random_state=42, n_jobs=-1
    )
    clf.fit(train_flats)
    raw  = clf.decision_function(train_flats)
    smin, smax = raw.min(), raw.max()

    def normalize_scores(r):
        return np.clip((-r - (-smax)) / max((-smin) - (-smax), 1e-9), 0, 1)

    scores = normalize_scores(clf.decision_function(test_flats))
    logger.info("✓ IF done | mean_score=%.4f", scores.mean())
    return scores


def run_lstm(train_wins, test_wins):
    logger.info("LSTM Autoencoder training (2-3 min lag sakta hai)...")
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, LSTM, Dense, RepeatVector, TimeDistributed
        )
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

        mdl.fit(
            train_wins, train_wins,
            epochs=30, batch_size=32, validation_split=0.1,
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
            verbose=1
        )

        pred   = mdl.predict(test_wins, verbose=0, batch_size=32)
        errors = np.mean(np.square(test_wins - pred), axis=(1, 2))
        scores = np.clip(
            (errors - errors.min()) / max(errors.max() - errors.min(), 1e-9),
            0, 1
        )
        logger.info("✓ LSTM done | mean_error=%.6f", errors.mean())
        return scores

    except Exception as e:
        logger.warning("LSTM failed: %s — using zero scores", e)
        return np.zeros(len(test_wins))


# ════════════════════════════════════════════════════════════════
# STEP 6 — Evaluate + print results
# ════════════════════════════════════════════════════════════════

def evaluate(y_true, y_pred_scores, model_name):
    y_pred  = (y_pred_scores >= THRESHOLD).astype(int)
    min_len = min(len(y_true), len(y_pred))
    y_true  = y_true[-min_len:]
    y_pred  = y_pred[-min_len:]

    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n{'='*58}")
    print(f"  {model_name}")
    print(f"{'='*58}")
    print(f"  Precision  : {p*100:.2f}%")
    print(f"  Recall     : {r*100:.2f}%")
    print(f"  F1-Score   : {f1*100:.2f}%")
    print(f"  FP Rate    : {fpr*100:.2f}%")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(classification_report(
        y_true, y_pred,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))

    return {
        "model": model_name,
        "precision": p, "recall": r,
        "f1": f1, "fpr": fpr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CloudSentinel — DB se seedha evaluation"
    )
    parser.add_argument(
        "--instance", default=None,
        help="Instance ID (e.g. my-windows-pc). Default: saare instances."
    )
    parser.add_argument(
        "--days", default=None, type=int,
        help=(
            "Kitne din ka data chahiye: "
            "--days 1 = aaj, --days 7 = pichle 7 din, --days 20 = pichle 20 din"
        )
    )
    parser.add_argument(
        "--start", default=None,
        help="Date range start YYYY-MM-DD (--end ke saath use karo)"
    )
    parser.add_argument(
        "--end", default=None,
        help="Date range end YYYY-MM-DD (--start ke saath use karo)"
    )
    parser.add_argument(
        "--output", default="db_evaluation_report.txt",
        help="Report file ka naam"
    )
    parser.add_argument(
        "--no-lstm", action="store_true",
        help="LSTM skip karo (fast evaluation ke liye)"
    )
    args = parser.parse_args()

    # ── Header ──────────────────────────────────────────────────
    print("\n" + "="*58)
    print("  CloudSentinel — Direct DB Evaluation")
    print("  MCA (AI/ML) | Chandigarh University")
    print("="*58)

    # ── Date range decide karo ──────────────────────────────────
    #
    #  3 modes:
    #  1. --days N       → aaj se N din peeche ka data
    #                      --days 1  = sirf aaj ka data
    #                      --days 20 = pichle 20 din ka data
    #
    #  2. --start & --end → exact date range beech ka data
    #                      --start 2026-05-09 --end 2026-05-24
    #
    #  3. kuch nahi       → poora data (saare records)
    #
    from datetime import timedelta

    start_dt = None
    end_dt   = None

    if args.days:
        # N din peeche se aaj tak
        end_dt   = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
        start_dt = (end_dt - timedelta(days=args.days - 1)).replace(
                       hour=0, minute=0, second=0, microsecond=0)
        print(f"\n  📅 Mode : Last {args.days} day(s)")
        print(f"  📅 From : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  📅 To   : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    elif args.start and args.end:
        # Exact date range
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(
                       hour=0, minute=0, second=0, microsecond=0)
        end_dt   = datetime.strptime(args.end,   "%Y-%m-%d").replace(
                       hour=23, minute=59, second=59, microsecond=0)
        print(f"\n  📅 Mode : Date range")
        print(f"  📅 From : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  📅 To   : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    elif args.start:
        # Sirf start diya — us din se aaj tak
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(
                       hour=0, minute=0, second=0, microsecond=0)
        end_dt   = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
        print(f"\n  📅 Mode : From {args.start} to today")

    else:
        # Kuch nahi — poora data
        print(f"\n  📅 Mode : All data (no date filter)")

    # ── Step 1: DB se data lo ───────────────────────────────────
    try:
        df = fetch_from_db(
            instance_id=args.instance,
            start_dt=start_dt,
            end_dt=end_dt
        )
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

    show_db_summary(df)

    # Minimum data check
    if len(df) < WINDOW_SIZE * 3:
        print(f"\n❌ Data bahut kam hai ({len(df)} rows).")
        print(f"   Kam se kam {WINDOW_SIZE * 3} rows chahiye evaluation ke liye.")
        sys.exit(1)

    # ── Step 2: Labels banao ────────────────────────────────────
    print("\n⏳ Z-score se anomaly labels generate ho rahe hain...")
    labels = generate_labels(df)
    print(f"   ✓ Labels: {labels.sum()} anomalies / {len(labels)} total "
          f"({labels.mean()*100:.1f}%)")

    # ── Step 3: Split + Normalize ───────────────────────────────
    df_tr, df_te, df_trn, df_ten, lbl_te = split_and_normalize(df, labels)

    # ── Step 4: Windows ─────────────────────────────────────────
    tr_w, tr_f, _      = make_windows(df_trn)
    te_w, te_f, te_idx = make_windows(df_ten)
    lbl_win            = lbl_te[te_idx]

    print(f"\n   Train windows : {len(tr_w)}")
    print(f"   Test windows  : {len(te_w)}")
    print(f"   Test anomalies: {lbl_win.sum()} ({lbl_win.mean()*100:.1f}%)")

    # ── Step 5: Z-score baseline ────────────────────────────────
    print("\n⏳ Z-score baseline evaluate ho raha hai...")
    base_full = np.zeros(len(df_te))
    valid_cols = [c for c in FEATURE_COLS if df_tr[c].std() > 0]
    for col in valid_cols:
        z         = np.abs((df_te[col].values - df_tr[col].mean()) / df_tr[col].std())
        base_full = np.maximum(base_full, z / 10.0)
    base = base_full[te_idx]
    r0   = evaluate(lbl_win, base, "Z-Score Baseline")

    # ── Step 6: Isolation Forest ────────────────────────────────
    print("\n⏳ Isolation Forest chal raha hai...")
    if_scores = run_isolation_forest(tr_f, te_f)
    r1        = evaluate(lbl_win, if_scores, "Isolation Forest")

    # ── Step 7: LSTM ────────────────────────────────────────────
    if not args.no_lstm:
        print("\n⏳ LSTM Autoencoder train ho raha hai...")
        lstm_scores = run_lstm(tr_w, te_w)
    else:
        print("\n⚠️  LSTM skip kiya (--no-lstm flag)")
        lstm_scores = np.zeros(len(te_w))
    r2 = evaluate(lbl_win, lstm_scores, "LSTM Autoencoder")

    # ── Step 8: Ensemble ────────────────────────────────────────
    print("\n⏳ Ensemble scores calculate ho rahe hain...")
    ensemble = (IF_WEIGHT * if_scores) + (LSTM_WEIGHT * lstm_scores)
    r3       = evaluate(
        lbl_win, ensemble,
        "CloudSentinel Ensemble (IF×0.4 + LSTM×0.6)"
    )

    # ── Final Summary Table ──────────────────────────────────────
    results = [r0, r1, r2, r3]
    print("\n" + "="*65)
    print("  📋 FINAL COMPARISON TABLE")
    print("="*65)
    print(f"  {'Model':<42} {'P':>6} {'R':>6} {'F1':>6} {'FPR':>6}")
    print("  " + "-"*61)
    for r in results:
        print(f"  {r['model']:<42} "
              f"{r['precision']*100:>5.1f}% "
              f"{r['recall']*100:>5.1f}% "
              f"{r['f1']*100:>5.1f}% "
              f"{r['fpr']*100:>5.1f}%")
    print("="*65)

    # Best model highlight
    best = max(results, key=lambda x: x["f1"])
    print(f"\n  🏆 Best F1: {best['model']} ({best['f1']*100:.2f}%)")

    # ── Report File Save ─────────────────────────────────────────
    lines = [
        "CloudSentinel — Direct DB Evaluation Report",
        "MCA (AI/ML) | Chandigarh University",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Instance  : {args.instance or 'All'}",
        f"Date range: {start_dt.strftime('%Y-%m-%d') if start_dt else 'All'} → {end_dt.strftime('%Y-%m-%d') if end_dt else 'All'}",
        f"Total rows: {len(df):,}",
        f"Train     : {len(df_tr)} | Test: {len(df_te)}",
        f"Anomalies : {lbl_win.sum()} in test set ({lbl_win.mean()*100:.1f}%)",
        "",
        f"{'Model':<44} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FPR':>8}",
        "-" * 74,
    ]
    for r in results:
        lines.append(
            f"{r['model']:<44} "
            f"{r['precision']*100:>9.2f}% "
            f"{r['recall']*100:>7.2f}% "
            f"{r['f1']*100:>7.2f}% "
            f"{r['fpr']*100:>7.2f}%"
        )

    with open(args.output, "w") as f:
        f.write("\n".join(lines))

    print(f"\n✅ Report saved: {args.output}")
    print("─"*65)


if __name__ == "__main__":
    main()
