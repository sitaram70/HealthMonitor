import time, requests, os, io
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Health Monitor MVP", layout="wide")
st.title("🩺 AI-Powered Health Monitoring — Dashboard")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Data Source")
    mode = st.radio("Mode", ["CSV Mode (local)", "API Mode (http)"], index=0)
    api_url = st.text_input("API base URL", "http://localhost:8000")
    st.markdown("---")
    st.header("Vitals Stream Controls")
    csv_path = st.text_input("Vitals CSV path", "data/vitals/vitals_train.csv")
    refresh = st.number_input("Refresh interval (sec)", value=0.2, min_value=0.05, step=0.05, format="%.2f")
    window = st.number_input("Window size (samples)", value=30, min_value=5, step=5)
    hr_spike = st.number_input("HR spike threshold (bpm)", value=120, min_value=60, step=1)
    fever = st.number_input("Fever threshold (°C)", value=37.8, min_value=36.0, step=0.1, format="%.1f")
    run_stream = st.toggle("Run stream", value=False)

# --- Tabs: Stream & Video Upload ---
tab1, tab2 = st.tabs(["📈 Vitals Stream", "🎥 Video Test (Upload)"])

# ------------- Tab 1: Vitals Stream -------------
with tab1:
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    plot_col, table_col = st.columns([2,1])

    if "alerts" not in st.session_state:
        st.session_state.alerts = []

    def compute_risk(hr, temp, hr_thr, temp_thr):
        score = 0
        if hr >= hr_thr: score += 1
        if temp >= temp_thr: score += 1
        if score >= 2: return "CRITICAL", 0.95
        if score == 1: return "ELEVATED", 0.65
        return "NORMAL", 0.15

    if mode.startswith("CSV"):
        try:
            df = pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            st.error(f"Could not read CSV at {csv_path}: {e}")
            st.stop()
        if "i" not in st.session_state:
            st.session_state.i = 0
        if run_stream:
            placeholder_plots = plot_col.empty()
            placeholder_table = table_col.empty()
            placeholder_k1 = col_kpi1.empty()
            placeholder_k2 = col_kpi2.empty()
            placeholder_k3 = col_kpi3.empty()
            while run_stream:
                i = st.session_state.i
                if i >= len(df):
                    st.success("End of stream reached.")
                    break
                window_df = df.iloc[max(0, i-window):i+1]
                row = df.iloc[i]
                hr = row["heart_rate_bpm"]
                temp = row["temp_c"]
                risk_label, risk_score = compute_risk(hr, temp, hr_spike, fever)
                placeholder_k1.metric("Heart Rate (bpm)", f"{hr:.1f}")
                placeholder_k2.metric("Temperature (°C)", f"{temp:.2f}")
                placeholder_k3.metric("Risk", risk_label)
                placeholder_plots.line_chart(window_df.set_index("timestamp")[["heart_rate_bpm", "temp_c"]])
                if risk_label == "CRITICAL":
                    st.session_state.alerts.append({
                        "timestamp": str(row["timestamp"]),
                        "heart_rate_bpm": float(hr),
                        "temp_c": float(temp),
                        "risk": risk_label
                    })
                if st.session_state.alerts:
                    placeholder_table.dataframe(pd.DataFrame(st.session_state.alerts))
                else:
                    placeholder_table.info("No alerts yet.")
                st.session_state.i += 1
                time.sleep(refresh)
        else:
            col_kpi1.metric("Heart Rate (bpm)", f"{df['heart_rate_bpm'].iloc[-1]:.1f}")
            col_kpi2.metric("Temperature (°C)", f"{df['temp_c'].iloc[-1]:.2f}")
            label, _ = compute_risk(df['heart_rate_bpm'].iloc[-1], df['temp_c'].iloc[-1], hr_spike, fever)
            col_kpi3.metric("Risk", label)
            plot_col.line_chart(df.set_index("timestamp")[["heart_rate_bpm", "temp_c"]])
            if st.session_state.alerts:
                table_col.dataframe(pd.DataFrame(st.session_state.alerts))
            else:
                table_col.info("No alerts yet.")
    else:
        # API Mode
        if st.button("Initialize API stream (/stream/reset)"):
            try:
                r = requests.post(f"{api_url}/stream/reset", params={"csv_path": csv_path}, timeout=10)
                st.write(r.json())
            except Exception as e:
                st.error(e)
        if run_stream:
            placeholder_plots = plot_col.empty()
            placeholder_table = table_col.empty()
            placeholder_k1 = col_kpi1.empty()
            placeholder_k2 = col_kpi2.empty()
            placeholder_k3 = col_kpi3.empty()
            history = []
            while run_stream:
                try:
                    r = requests.get(f"{api_url}/stream/next", params={"hr_spike": hr_spike, "fever": fever}, timeout=10)
                    js = r.json()
                except Exception as e:
                    st.error(e); break
                if js.get("end"):
                    st.success("End of stream from API."); break
                if "row" in js:
                    row = js["row"]
                    history.append(row)
                    dfh = pd.DataFrame(history)
                    placeholder_k1.metric("Heart Rate (bpm)", f"{row['heart_rate_bpm']:.1f}")
                    placeholder_k2.metric("Temperature (°C)", f"{row['temp_c']:.2f}")
                    placeholder_k3.metric("Risk", row['label'])
                    dfh["timestamp"] = pd.to_datetime(dfh["timestamp"])
                    placeholder_plots.line_chart(dfh.set_index("timestamp")[["heart_rate_bpm", "temp_c"]])
                    # alerts
                    try:
                        ar = requests.get(f"{api_url}/alerts", timeout=5).json().get("alerts", [])
                        if ar:
                            placeholder_table.dataframe(pd.DataFrame(ar))
                        else:
                            placeholder_table.info("No alerts yet.")
                    except:
                        pass
                time.sleep(refresh)

# ------------- Tab 2: Video Upload -------------
with tab2:
    st.subheader("Upload a short video clip (.mp4/.mov/.avi/.mkv)")
    uploaded = st.file_uploader("Choose a file", type=["mp4","mov","avi","mkv"])

    st.markdown("#### Video detection thresholds")
    c1, c2, c3 = st.columns(3)
    with c1:
        mean_thresh = st.number_input("Mean threshold", value=0.45, min_value=0.0, max_value=1.0, step=0.01, key="mean_thresh")
    with c2:
        peak_thresh = st.number_input("Peak (pXX) threshold", value=0.60, min_value=0.0, max_value=1.0, step=0.01, key="peak_thresh")
    with c3:
        peak_pct = st.number_input("Peak percentile", value=95, min_value=50, max_value=100, step=1, key="peak_pct")

    run_predict = st.button("Analyze Video", disabled=uploaded is None)

    if uploaded and run_predict:
        if mode.startswith("API"):
            # Build multipart request: file + form fields
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")}
            data = {
                "mean_thresh": str(mean_thresh),
                "peak_thresh": str(peak_thresh),
                "peak_pct": str(int(peak_pct)),
            }
            try:
                resp = requests.post(f"{api_url}/predict/video", files=files, data=data, timeout=120)
                resp.raise_for_status()
                js = resp.json()

                with st.expander("Raw response"):
                    st.json(js)

                label = js.get("label", "normal").upper()
                score = js.get("score", 0.0)
                st.metric("Prediction", f"{label}", f"{score:.2f}")

                # Echo the params actually used by the API (if returned)
                p = js.get("params") or {}
                st.caption(
                    f"mean_thresh={p.get('mean_thresh', mean_thresh)} | "
                    f"peak_thresh={p.get('peak_thresh', peak_thresh)} | "
                    f"peak_pct={p.get('peak_pct', peak_pct)}"
                )
            except Exception as e:
                st.error(f"Request failed: {e}")
        else:
            st.info("Switch to API Mode to use backend inference. (Local inference path can be added if needed.)")
