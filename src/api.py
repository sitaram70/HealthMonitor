#!/usr/bin/env python3
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, io, tempfile, pandas as pd, numpy as np
from datetime import datetime
from .inference import load_vitals_model, vitals_predict_proba, video_predict

app = FastAPI(title="Health Monitor API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory "stream" and alerts
STREAM_DF = None
STREAM_I = 0
ALERTS : List[Dict[str,Any]] = []

@app.get("/health")
def health():
    return {"status":"ok","time": datetime.utcnow().isoformat()}

class VitalsSample(BaseModel):
    heart_rate_bpm: float
    temp_c: float
    hr_roll_mean: Optional[float] = None
    temp_roll_mean: Optional[float] = None

@app.post("/predict")
def predict_vitals(vitals: VitalsSample):
    X = np.array([[
        vitals.heart_rate_bpm,
        vitals.temp_c,
        vitals.hr_roll_mean if vitals.hr_roll_mean is not None else vitals.heart_rate_bpm,
        vitals.temp_roll_mean if vitals.temp_roll_mean is not None else vitals.temp_c,
    ]])
    model = load_vitals_model()
    proba = vitals_predict_proba(model, X)[0].item()
    label = "CRITICAL" if proba>=0.9 else ("ELEVATED" if proba>=0.5 else "NORMAL")
    return {"label": label, "score": proba}

@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    mean_thresh: Optional[float] = Form(None),
    peak_thresh: Optional[float] = Form(None),
    peak_pct: Optional[int] = Form(None),
):
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in [".mp4",".avi",".mov",".mkv"]:
        return {"error": "Unsupported file type. Upload a video (.mp4/.avi/.mov/.mkv)"}
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        result = video_predict(
            tmp_path,
            mean_thresh=mean_thresh,
            peak_thresh=peak_thresh,
            peak_pct=peak_pct
        )
        return result
    finally:
        try: os.remove(tmp_path)
        except: pass

@app.post("/stream/reset")
def stream_reset(csv_path: Optional[str] = None):
    global STREAM_DF, STREAM_I, ALERTS
    if csv_path is None:
        csv_path = "data/vitals/vitals_train.csv"
    if not os.path.exists(csv_path):
        return {"error": f"CSV not found at {csv_path}"}
    STREAM_DF = pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    STREAM_I = 0
    ALERTS = []
    return {"ok": True, "rows": len(STREAM_DF)}

@app.get("/stream/next")
def stream_next(hr_spike: float = 120.0, fever: float = 37.8):
    global STREAM_DF, STREAM_I, ALERTS
    if STREAM_DF is None:
        return {"error":"Stream not initialized. Call /stream/reset first."}
    if STREAM_I >= len(STREAM_DF):
        return {"end": True}
    row = STREAM_DF.iloc[STREAM_I]
    STREAM_I += 1
    hr = float(row["heart_rate_bpm"]); temp = float(row["temp_c"])
    score = 0
    if hr >= hr_spike: score += 1
    if temp >= fever: score += 1
    label = "CRITICAL" if score>=2 else ("ELEVATED" if score==1 else "NORMAL")
    item = {"timestamp": str(row["timestamp"]), "heart_rate_bpm": hr, "temp_c": temp, "label": label}
    if label=="CRITICAL":
        ALERTS.append(item)
    return {"row": item, "i": STREAM_I}

@app.get("/alerts")
def alerts():
    return {"alerts": ALERTS}
