#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import time
import random
from pathlib import Path

st.set_page_config(page_title="AI Health Monitor Dashboard", layout="wide")

st.title("🩺 AI‑Powered Health Monitoring — Demo Dashboard")

# Sidebar controls
mode = st.sidebar.radio("Data Source", ["Simulated", "CSV Upload"])
refresh_rate = st.sidebar.slider("Refresh every (sec)", 1, 10, 3)

placeholder = st.empty()

# Simulated or uploaded vitals
if mode == "CSV Upload":
    file = st.sidebar.file_uploader("Upload vitals CSV", type="csv")
    if file:
        df = pd.read_csv(file, parse_dates=["timestamp"])
else:
    # simulate streaming vitals (heart rate and temp)
    df = pd.DataFrame(columns=["timestamp","heart_rate_bpm","temp_c","risk_event"])
    now = pd.Timestamp.now()
    for i in range(100):
        hr = 70 + random.randint(-5,5)
        temp = 36.8 + random.random()*0.4
        risk = 1 if hr > 100 or temp > 37.8 else 0
        df.loc[len(df)] = [now+pd.Timedelta(seconds=i), hr, temp, risk]

with placeholder.container():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Heart Rate (bpm)")
        st.line_chart(df.set_index("timestamp")["heart_rate_bpm"])
    with col2:
        st.subheader("Temperature (°C)")
        st.line_chart(df.set_index("timestamp")["temp_c"])
    st.subheader("Risk Events")
    st.bar_chart(df.set_index("timestamp")["risk_event"])

    # Latest reading + status
    last = df.iloc[-1]
    status = "🟢 Normal"
    if last["risk_event"] == 1:
        status = "🔴 ALERT!"
    st.metric("Current Status", status, help="Based on latest vitals & model logic")

st.info("This is a starter dashboard. In Milestone 4 (Week 13) students should connect real model predictions and video-based fall detection alerts here.")
