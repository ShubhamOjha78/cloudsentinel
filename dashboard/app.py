"""
CloudSentinel – dashboard/app.py
==================================================
Real-time monitoring dashboard built with Streamlit.

Sections:
  1. Sidebar  – instance selector, time range, threshold
  2. Header   – system status indicator
  3. Metrics  – 4 live time-series charts (CPU, Mem, Disk, Net)
  4. Scores   – ensemble anomaly score timeline
  5. Events   – recent anomaly events table
  6. Heatmap  – anomaly frequency by day-of-week × hour

Run with:
    streamlit run dashboard/app.py --server.port 8501
"""

import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow project-root imports (works on both Windows and Linux)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.db_utils import (
    fetch_metrics_range,
    fetch_recent_anomalies,
    get_distinct_instances,
    check_db_connection,
    DB_CONFIG,
)

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title = "CloudSentinel – Anomaly Dashboard",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ------------------------------------------------------------------
# CSS overrides
# ------------------------------------------------------------------
st.markdown("""
<style>
  .main { background-color: #F5F8FF; }
  .status-normal  { background:#27AE60; color:#fff; padding:6px 14px;
                    border-radius:20px; font-weight:bold; display:inline-block; }
  .status-alert   { background:#C0392B; color:#fff; padding:6px 14px;
                    border-radius:20px; font-weight:bold; display:inline-block; }
  .metric-card    { background:#fff; padding:16px; border-radius:10px;
                    border-left:4px solid #2E75B6; margin-bottom:8px; }
  h1 { color: #FFFFFF !important; }
  h2, h3 { color: #60B4FF !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# ⚡ DATABASE HEALTH CHECK — show setup guide if DB not reachable
# ------------------------------------------------------------------
db_ok, db_err = check_db_connection()

if not db_ok:
    st.title("🛡️ CloudSentinel")
    st.error("**PostgreSQL database is not reachable.** The dashboard cannot load without it.", icon="🔌")

    st.markdown(f"""
    **Connection details being used:**
    ```
    Host     : {DB_CONFIG['host']}
    Port     : {DB_CONFIG['port']}
    Database : {DB_CONFIG['dbname']}
    User     : {DB_CONFIG['user']}
    ```
    **Error:** `{db_err}`
    """)

    st.info("### 🪟 Windows Setup Guide — Fix in 3 steps", icon="ℹ️")

    with st.expander("**Step 1 — Install PostgreSQL (if not installed)**", expanded=True):
        st.markdown("""
        1. Download from **https://www.postgresql.org/download/windows/**
        2. Run the installer — remember the **password** you set for `postgres` user
        3. Default port is **5432** — keep it as is
        4. After install, PostgreSQL service starts automatically
        """)

    with st.expander("**Step 2 — Create the database**", expanded=True):
        st.markdown("""
        Open **pgAdmin** or **SQL Shell (psql)** from Start Menu and run:
        ```sql
        CREATE DATABASE cloudsentinel;
        ```
        Then run the schema file:
        ```
        psql -U postgres -d cloudsentinel -f D:\\cloudsentinel\\database\\schema.sql
        ```
        """)

    with st.expander("**Step 3 — Configure config.env**", expanded=True):
        st.markdown("""
        Create a file named **`config.env`** in `D:\\cloudsentinel\\` with:
        ```
        DB_HOST=localhost
        DB_PORT=5432
        DB_NAME=cloudsentinel
        DB_USER=postgres
        DB_PASSWORD=your_postgres_password_here
        ```
        Then restart the dashboard:
        ```
        streamlit run dashboard/app.py
        ```
        """)

    with st.expander("**Step 4 — Start collecting data**"):
        st.markdown("""
        Open a new terminal and run the data collection agent:
        ```
        cd D:\\cloudsentinel
        python agents\\collector.py --instance-id my-windows-pc --interval 30
        ```
        Wait a few minutes, then refresh this dashboard.
        """)

    st.markdown("---")
    if st.button("🔄 Retry Connection"):
        st.rerun()

    st.stop()   # ← Do not render the rest of the dashboard

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
st.sidebar.markdown("## <span style='color:#FFFFFF; font-weight:700'>🛡️ CloudSentinel</span>", unsafe_allow_html=True)
st.sidebar.markdown("<span style='color:#AACCFF'>Real-Time Anomaly Detection</span>", unsafe_allow_html=True)
st.sidebar.divider()

# Instance selector
try:
    instances = get_distinct_instances()
except Exception:
    instances = []
if not instances:
    instances = ["No data yet — run collector.py first"]
selected_instance = st.sidebar.selectbox("📡 Cloud Instance", instances)

# Time range
time_options = {"Last 1 hour": 1, "Last 6 hours": 6,
                "Last 24 hours": 24, "Last 7 days": 168}
time_label   = st.sidebar.selectbox("⏱️ Time Range", list(time_options.keys()))
hours_back   = time_options[time_label]

# Anomaly threshold slider
threshold = st.sidebar.slider(
    "🎚️ Anomaly Threshold", min_value=0.50, max_value=0.95,
    value=0.65, step=0.01
)

# Auto-refresh
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=True)

st.sidebar.divider()
st.sidebar.caption("MCA (AI/ML) Project ")

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
@st.cache_data(ttl=30)
def load_metrics(instance_id: str, hours: int) -> pd.DataFrame:
    try:
        end   = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)
        records = fetch_metrics_range(instance_id, start, end)
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
        return df.sort_values("timestamp")
    except Exception as exc:
        st.warning(f"Could not load metrics: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_anomalies(instance_id: str, hours: int = 1) -> pd.DataFrame:
    try:
        from datetime import datetime, timezone, timedelta
        from database.db_utils import fetch_metrics_range
        end   = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)
        records = fetch_recent_anomalies(instance_id=instance_id, limit=500)
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["detected_at"] = pd.to_datetime(df["detected_at"], utc=True).dt.tz_convert("Asia/Kolkata")
        cutoff = pd.Timestamp.now(tz="Asia/Kolkata") - pd.Timedelta(hours=hours)
        df = df[df["detected_at"] >= cutoff]
        return df.sort_values("detected_at", ascending=False)
    except Exception as exc:
        st.warning(f"Could not load anomaly events: {exc}")
        return pd.DataFrame()


df_metrics   = load_metrics(selected_instance, hours_back)
df_anomalies = load_anomalies(selected_instance,  hours=hours_back)

# Recent anomalies for alert status
recent_anomalies = df_anomalies[
    df_anomalies["detected_at"] >= datetime.now(timezone.utc) - timedelta(hours=1)
] if not df_anomalies.empty else pd.DataFrame()

has_recent_alert = not recent_anomalies.empty

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
col_title, col_status, col_refresh = st.columns([4, 2, 1])

with col_title:
    st.markdown("## <span style='color:#FFFFFF; font-size:2rem; font-weight:700'>🛡️ CloudSentinel Monitoring Dashboard</span>", unsafe_allow_html=True)
    st.caption(f"Instance: **{selected_instance}** | Range: {time_label}")

with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    if has_recent_alert:
        severity = recent_anomalies.iloc[0]["severity"]
        st.markdown(
            f'<span class="status-alert">⚠️ {severity} ALERT</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-normal">✅ NORMAL</span>',
            unsafe_allow_html=True
        )

with col_refresh:
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

st.divider()

# ------------------------------------------------------------------
# Summary KPI cards
# ------------------------------------------------------------------
if not df_metrics.empty:
    latest = df_metrics.iloc[-1]
    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("CPU Usage",    f"{latest['cpu_percent']:.1f}%",
              delta=f"{latest['cpu_percent'] - df_metrics['cpu_percent'].mean():.1f}% vs avg")
    k2.metric("Memory Usage", f"{latest['mem_percent']:.1f}%",
              delta=f"{latest['mem_percent'] - df_metrics['mem_percent'].mean():.1f}% vs avg")
    k3.metric("Disk Read",    f"{latest['disk_read_mb']:.2f} MB/s")
    k4.metric("Net Recv",     f"{latest['net_recv_kb']:.1f} KB/s")
    k5.metric("Anomalies (1h)", len(recent_anomalies))

    st.divider()

# ------------------------------------------------------------------
# Helper: add anomaly markers to a plotly figure
# ------------------------------------------------------------------
def add_anomaly_markers(fig, df_a: pd.DataFrame, y_max: float) -> None:
    if df_a.empty:
        return
    for _, row in df_a.iterrows():
        color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "gold"}.get(
            row.get("severity", "LOW"), "red"
        )
        fig.add_vrect(
            x0=row["detected_at"], x1=row["detected_at"],
            line_width=1.5, line_color=color,
            annotation_text=row.get("severity", ""),
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color=color,
        )


# ------------------------------------------------------------------
# Section 1: Resource Metrics Charts
# ------------------------------------------------------------------
st.subheader("📊 Real-Time Resource Metrics")

if df_metrics.empty:
    st.info("No metrics data available for the selected instance and time range.")
else:
    # Filter anomalies to the same time range
    if not df_anomalies.empty:
        start_ts = df_metrics["timestamp"].min()
        df_anom_range = df_anomalies[df_anomalies["detected_at"] >= start_ts]
    else:
        df_anom_range = pd.DataFrame()

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # CPU
    with row1_col1:
        fig_cpu = px.line(df_metrics, x="timestamp", y="cpu_percent",
                          title="CPU Utilization (%)",
                          color_discrete_sequence=["#2E75B6"])
        fig_cpu.update_layout(yaxis_range=[0, 100], height=280, margin=dict(t=40, b=20))
        add_anomaly_markers(fig_cpu, df_anom_range, 100)
        st.plotly_chart(fig_cpu, width='stretch')

    # Memory
    with row1_col2:
        fig_mem = px.line(df_metrics, x="timestamp", y="mem_percent",
                          title="Memory Utilization (%)",
                          color_discrete_sequence=["#8E44AD"])
        fig_mem.update_layout(yaxis_range=[0, 100], height=280, margin=dict(t=40, b=20))
        add_anomaly_markers(fig_mem, df_anom_range, 100)
        st.plotly_chart(fig_mem, width='stretch')

    # Disk I/O
    with row2_col1:
        fig_disk = go.Figure()
        fig_disk.add_trace(go.Scatter(
            x=df_metrics["timestamp"], y=df_metrics["disk_read_mb"],
            name="Read MB/s", line=dict(color="#27AE60")
        ))
        fig_disk.add_trace(go.Scatter(
            x=df_metrics["timestamp"], y=df_metrics["disk_write_mb"],
            name="Write MB/s", line=dict(color="#E74C3C")
        ))
        fig_disk.update_layout(title="Disk I/O (MB/s)", height=280,
                               margin=dict(t=40, b=20), legend=dict(orientation="h"))
        add_anomaly_markers(fig_disk, df_anom_range,
                            df_metrics[["disk_read_mb","disk_write_mb"]].max().max())
        st.plotly_chart(fig_disk, width='stretch')

    # Network I/O
    with row2_col2:
        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(
            x=df_metrics["timestamp"], y=df_metrics["net_recv_kb"],
            name="Recv KB/s", line=dict(color="#F39C12")
        ))
        fig_net.add_trace(go.Scatter(
            x=df_metrics["timestamp"], y=df_metrics["net_sent_kb"],
            name="Sent KB/s", line=dict(color="#1ABC9C")
        ))
        fig_net.update_layout(title="Network I/O (KB/s)", height=280,
                              margin=dict(t=40, b=20), legend=dict(orientation="h"))
        add_anomaly_markers(fig_net, df_anom_range,
                            df_metrics[["net_recv_kb","net_sent_kb"]].max().max())
        st.plotly_chart(fig_net, width='stretch')

# ------------------------------------------------------------------
# Section 2: Anomaly Score Timeline
# ------------------------------------------------------------------
st.subheader("🎯 Anomaly Score Timeline")

if not df_anomalies.empty:
    fig_score = go.Figure()

    fig_score.add_trace(go.Scatter(
        x    = df_anomalies["detected_at"],
        y    = df_anomalies["ensemble_score"],
        mode = "lines+markers",
        name = "Ensemble Score",
        line = dict(color="#2E75B6", width=2),
        marker = dict(
            size  = 8,
            color = df_anomalies["severity"].map(
                {"HIGH": "#C0392B", "MEDIUM": "#E67E22",
                 "LOW": "#F1C40F", "NORMAL": "#27AE60"}
            ).fillna("#2E75B6"),
        ),
    ))

    # IF score trace
    fig_score.add_trace(go.Scatter(
        x    = df_anomalies["detected_at"],
        y    = df_anomalies["if_score"],
        mode = "lines",
        name = "Isolation Forest",
        line = dict(color="#8E44AD", width=1, dash="dot"),
        opacity = 0.7,
    ))

    # LSTM trace
    fig_score.add_trace(go.Scatter(
        x    = df_anomalies["detected_at"],
        y    = df_anomalies["lstm_error"],
        mode = "lines",
        name = "LSTM Error",
        line = dict(color="#27AE60", width=1, dash="dash"),
        opacity = 0.7,
    ))

    # Threshold line
    fig_score.add_hline(
        y             = threshold,
        line_dash     = "solid",
        line_color    = "red",
        line_width    = 1.5,
        annotation_text  = f"Threshold ({threshold:.2f})",
        annotation_position = "top right",
    )

    fig_score.update_layout(
        height     = 320,
        yaxis      = dict(range=[0, 1.05], title="Anomaly Score"),
        xaxis_title= "Time",
        legend     = dict(orientation="h"),
        margin     = dict(t=20, b=20),
    )
    st.plotly_chart(fig_score, width='stretch')
else:
    st.info("No anomaly score history available yet.")

# ------------------------------------------------------------------
# Section 3: Recent Anomaly Events Table
# ------------------------------------------------------------------
st.subheader("🚨 Recent Anomaly Events")

if not df_anomalies.empty:
    display_cols = [
        "detected_at", "instance_id", "ensemble_score",
        "if_score", "lstm_error", "severity", "alert_sent"
    ]
    display_df = df_anomalies[
        [c for c in display_cols if c in df_anomalies.columns]
    ].head(30).copy()

    # Format timestamps
    if "detected_at" in display_df.columns:
        display_df["detected_at"] = display_df["detected_at"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    # Round scores
    for col in ["ensemble_score", "if_score", "lstm_error"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)

    def highlight_severity(row):
        colors = {
            "HIGH":   "background-color: #7B241C; color: white",
            "MEDIUM": "background-color: #784212; color: white",
            "LOW":    "background-color: #7D6608; color: white",
        }
        color = colors.get(row.get("severity", ""), "color: white")
        return [color] * len(row)

    styled = display_df.style.apply(highlight_severity, axis=1)
    st.dataframe(styled, width='stretch', height=300)

    # Export button
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label    = "📥 Export to CSV",
        data     = csv,
        file_name= f"anomalies_{selected_instance}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime     = "text/csv",
    )
else:
    st.success("✅ No anomalies detected for this instance.")

# ------------------------------------------------------------------
# Section 4: Anomaly Frequency Heatmap
# ------------------------------------------------------------------
st.subheader("🗓️ Anomaly Frequency Heatmap (Day × Hour)")

if not df_anomalies.empty and len(df_anomalies) >= 5:
    df_heat = df_anomalies.copy()
    df_heat["hour"]       = df_heat["detected_at"].dt.hour
    df_heat["day_of_week"]= df_heat["detected_at"].dt.day_name()

    pivot = (
        df_heat.groupby(["day_of_week", "hour"])
        .size()
        .reset_index(name="count")
    )

    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot["day_of_week"] = pd.Categorical(
        pivot["day_of_week"], categories=day_order, ordered=True
    )

    fig_heat = px.density_heatmap(
        pivot,
        x         = "hour",
        y         = "day_of_week",
        z         = "count",
        color_continuous_scale = "Reds",
        title     = "Anomaly Frequency by Day and Hour",
        labels    = {"hour": "Hour of Day", "day_of_week": "Day", "count": "Anomalies"},
    )
    fig_heat.update_layout(height=320, margin=dict(t=40, b=20))
    st.plotly_chart(fig_heat, width='stretch')
else:
    st.info("Need at least 5 anomaly events to render the heatmap.")

# ------------------------------------------------------------------
# Section 5: CSV Upload & Analysis (Auto-Train on CSV data)
# ------------------------------------------------------------------
st.divider()
st.subheader("📂 Cloud Metrics CSV Analysis")
st.caption("CSV upload karo → Model **usi CSV ke data pe** train hoga → Phir anomalies detect karega (accurate results)")

st.info("""
**Kaise kaam karta hai:**
- CSV ka **pehla 80% data** → Model training (normal behavior seekhna)
- CSV ka **aakhri 20% data** → Anomaly detection (test)
- Har user ke CSV pe alag model train hota hai → Accurate results ✅
""")

uploaded_file = st.file_uploader(
    "CSV file choose karo",
    type=["csv"],
    help="Supported columns: timestamp, cpu_usage/cpu_percent, memory_usage/mem_percent, disk_usage/disk_read_mb, etc."
)

if uploaded_file is not None:
    import joblib
    from pathlib import Path
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import IsolationForest

    # ---- Constants ----
    COL_MAP = {
        "cpu_usage": "cpu_percent", "cpu_utilization": "cpu_percent", "cpu": "cpu_percent",
        "memory_usage": "mem_percent", "memory_utilization": "mem_percent",
        "mem_usage": "mem_percent", "memory": "mem_percent",
        "disk_usage": "disk_read_mb", "disk_read": "disk_read_mb",
        "disk_write": "disk_write_mb",
        "network_in": "net_recv_kb", "net_recv": "net_recv_kb",
        "network_out": "net_sent_kb", "net_sent": "net_sent_kb",
    }
    FEATURE_COLS = ["cpu_percent","mem_percent","disk_read_mb","disk_write_mb","net_recv_kb","net_sent_kb"]
    WINDOW_SIZE  = 60
    IF_WEIGHT    = 0.40
    LSTM_WEIGHT  = 0.60
    THRESHOLD    = 0.65
    TRAIN_RATIO  = 0.80   # 80% train, 20% test

    try:
        # ---- Step 1: Load & map columns ----
        df_csv = pd.read_csv(uploaded_file)
        df_csv.columns = [c.strip().lower() for c in df_csv.columns]
        df_csv = df_csv.rename(columns=COL_MAP)

        missing_cols = []
        for col in FEATURE_COLS:
            if col not in df_csv.columns:
                df_csv[col] = 0.0
                missing_cols.append(col)

        if "timestamp" in df_csv.columns:
            df_csv["timestamp"] = pd.to_datetime(df_csv["timestamp"])
            df_csv = df_csv.sort_values("timestamp").reset_index(drop=True)

        df_csv[FEATURE_COLS] = df_csv[FEATURE_COLS].ffill().bfill()

        # ---- Step 2: Show data info ----
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", len(df_csv))
        train_size = int(len(df_csv) * TRAIN_RATIO)
        test_size  = len(df_csv) - train_size
        c2.metric("Training Rows (80%)", train_size)
        c3.metric("Testing Rows (20%)", test_size)
        c4.metric("Missing cols (→0)", len(missing_cols))

        if missing_cols:
            st.info(f"ℹ️ Missing columns — 0 se fill kiye: `{'`, `'.join(missing_cols)}`")

        with st.expander("📋 CSV Preview (first 5 rows)"):
            preview_cols = (["timestamp"] if "timestamp" in df_csv.columns else []) + FEATURE_COLS
            st.dataframe(df_csv[preview_cols].head())

        if len(df_csv) < WINDOW_SIZE * 2:
            st.error(f"❌ Data bahut kam hai — kam se kam {WINDOW_SIZE * 2} rows chahiye. Aapki CSV: {len(df_csv)} rows.")
        else:
            # ---- Step 3: Train/Test split ----
            df_train = df_csv.iloc[:train_size].copy().reset_index(drop=True)
            df_test  = df_csv.iloc[train_size:].copy().reset_index(drop=True)

            # ---- Step 4: Normalize on TRAIN data only ----
            scaler = MinMaxScaler()
            scaler.fit(df_train[FEATURE_COLS])

            df_train_norm = df_train.copy()
            df_test_norm  = df_test.copy()
            df_train_norm[FEATURE_COLS] = scaler.transform(df_train[FEATURE_COLS])
            df_test_norm[FEATURE_COLS]  = scaler.transform(df_test[FEATURE_COLS])

            # ---- Step 5: Build windows ----
            def make_windows(df_norm):
                data = df_norm[FEATURE_COLS].values.astype("float32")
                wins, flts, idxs = [], [], []
                for i in range(WINDOW_SIZE, len(data) + 1):
                    wins.append(data[i - WINDOW_SIZE : i])
                    flts.append(data[i - 1])
                    idxs.append(i - 1)
                return np.array(wins), np.array(flts), idxs

            train_wins, train_flats, _ = make_windows(df_train_norm)
            test_wins,  test_flats,  test_idx = make_windows(df_test_norm)

            if len(test_wins) == 0:
                st.error("❌ Test data bahut kam hai — badi CSV upload karo.")
            else:
                with st.spinner("⚙️ Step 1/3 — Isolation Forest train ho raha hai CSV data pe..."):

                    # ---- Step 6: Train Isolation Forest on CSV ----
                    clf = IsolationForest(
                        n_estimators=100, contamination=0.05,
                        random_state=42, n_jobs=-1
                    )
                    clf.fit(train_flats)

                    raw_train = clf.decision_function(train_flats)
                    s_min = float(raw_train.min())
                    s_max = float(raw_train.max())

                    def if_normalize(raw):
                        inv   = -raw
                        denom = max((-s_min) - (-s_max), 1e-9)
                        return np.clip((inv - (-s_max)) / denom, 0.0, 1.0)

                    if_scores = if_normalize(clf.decision_function(test_flats))

                with st.spinner("⚙️ Step 2/3 — LSTM Autoencoder train ho raha hai CSV data pe (2-3 min)..."):

                    # ---- Step 7: Train LSTM on CSV ----
                    lstm_scores = np.zeros(len(test_wins))
                    try:
                        import tensorflow as tf
                        from tensorflow.keras.models import Model  # type: ignore
                        from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed  # type: ignore
                        from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

                        inp = Input(shape=(WINDOW_SIZE, len(FEATURE_COLS)))
                        x   = LSTM(64, return_sequences=True)(inp)
                        x   = LSTM(32, return_sequences=False)(x)
                        x   = RepeatVector(WINDOW_SIZE)(x)
                        x   = LSTM(32, return_sequences=True)(x)
                        x   = LSTM(64, return_sequences=True)(x)
                        out = TimeDistributed(Dense(len(FEATURE_COLS)))(x)

                        csv_lstm = Model(inputs=inp, outputs=out)
                        csv_lstm.compile(optimizer="adam", loss="mse")

                        csv_lstm.fit(
                            train_wins, train_wins,
                            epochs=30, batch_size=32,
                            validation_split=0.1,
                            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
                            verbose=0
                        )

                        # Score on test
                        pred       = csv_lstm.predict(test_wins, verbose=0, batch_size=32)
                        errors     = np.mean(np.square(test_wins - pred), axis=(1, 2))
                        e_min, e_max = errors.min(), errors.max()
                        denom      = max(e_max - e_min, 1e-9)
                        lstm_scores = np.clip((errors - e_min) / denom, 0.0, 1.0)

                    except Exception as e:
                        st.warning(f"⚠️ LSTM training failed: {e} — sirf Isolation Forest use hoga")
                        lstm_scores = if_scores.copy()

                with st.spinner("⚙️ Step 3/3 — Ensemble scoring..."):

                    # ---- Step 8: Ensemble ----
                    ensemble = (IF_WEIGHT * if_scores) + (LSTM_WEIGHT * lstm_scores)
                    sev_fn   = lambda s: "HIGH" if s>=0.88 else "MEDIUM" if s>=0.75 else "LOW" if s>=0.65 else "NORMAL"
                    severities = [sev_fn(s) for s in ensemble]

                # ---- Step 9: Results ----
                result_df = df_test.iloc[test_idx].copy().reset_index(drop=True)
                result_df["if_score"]       = np.round(if_scores, 4)
                result_df["lstm_score"]     = np.round(lstm_scores, 4)
                result_df["ensemble_score"] = np.round(ensemble, 4)
                result_df["severity"]       = severities
                result_df["is_anomaly"]     = ensemble >= THRESHOLD

                anomalies_df = result_df[result_df["is_anomaly"]]

                st.success("✅ Analysis complete! Model CSV ke data pe train hua — results accurate hain.")

                # ---- Summary ----
                st.markdown("#### 🎯 Detection Results")
                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric("Test Windows", len(result_df))
                r2.metric("Anomalies", len(anomalies_df))
                r3.metric("HIGH 🔴", len(anomalies_df[anomalies_df["severity"]=="HIGH"]))
                r4.metric("MEDIUM 🟠", len(anomalies_df[anomalies_df["severity"]=="MEDIUM"]))
                r5.metric("LOW 🟡", len(anomalies_df[anomalies_df["severity"]=="LOW"]))

                # ---- Score chart ----
                x_axis = result_df["timestamp"] if "timestamp" in result_df.columns else result_df.index
                fig_csv = go.Figure()
                fig_csv.add_trace(go.Scatter(x=x_axis, y=result_df["ensemble_score"],
                    mode="lines", name="Ensemble Score", line=dict(color="#2E75B6", width=2)))
                fig_csv.add_trace(go.Scatter(x=x_axis, y=result_df["if_score"],
                    mode="lines", name="Isolation Forest", line=dict(color="#8E44AD", width=1, dash="dot"), opacity=0.7))
                fig_csv.add_trace(go.Scatter(x=x_axis, y=result_df["lstm_score"],
                    mode="lines", name="LSTM Score", line=dict(color="#27AE60", width=1, dash="dash"), opacity=0.7))
                fig_csv.add_hline(y=THRESHOLD, line_color="red", line_width=1.5,
                    annotation_text=f"Threshold ({THRESHOLD})", annotation_position="top right")
                fig_csv.update_layout(title="Anomaly Score Timeline — CSV Data (Test Set)",
                    height=350, yaxis=dict(range=[0,1.05]),
                    legend=dict(orientation="h"), margin=dict(t=40,b=20))
                st.plotly_chart(fig_csv, width='stretch')

                # ---- CPU chart ----
                if "cpu_percent" in result_df.columns:
                    fig_cpu_csv = px.line(result_df, x=x_axis, y="cpu_percent",
                        title="CPU Utilization (%) — Anomaly markers ke saath",
                        color_discrete_sequence=["#2E75B6"])
                    fig_cpu_csv.update_layout(height=250, margin=dict(t=40, b=20))
                    for _, row in anomalies_df.iterrows():
                        clr = {"HIGH":"red","MEDIUM":"orange","LOW":"gold"}.get(row["severity"],"red")
                        xv  = row["timestamp"] if "timestamp" in row.index else row.name
                        fig_cpu_csv.add_vrect(x0=xv, x1=xv, line_width=1, line_color=clr)
                    st.plotly_chart(fig_cpu_csv, width='stretch')

                # ---- Anomaly table ----
                if not anomalies_df.empty:
                    st.markdown("#### 🚨 Detected Anomaly Events")
                    show_cols = [c for c in ["timestamp","cpu_percent","mem_percent",
                                             "ensemble_score","if_score","lstm_score","severity"]
                                 if c in anomalies_df.columns]
                    st.dataframe(anomalies_df[show_cols].head(50), width='stretch', height=300)

                # ---- Download ----
                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Results CSV Download karo",
                    data=csv_out,
                    file_name="anomaly_results.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)

# ------------------------------------------------------------------
# Auto-refresh
# ------------------------------------------------------------------
if auto_refresh:
    time.sleep(30)
    st.cache_data.clear()
    st.rerun()
