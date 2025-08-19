#!/usr/bin/env python3
import os, glob, io, cv2, numpy as np, pandas as pd, joblib, tempfile, shutil
from typing import Tuple, Dict, Any
from typing import Optional

# Event-aware thresholds (override with env vars if you want)
VIDEO_MEAN_THRESH = float(os.getenv("VIDEO_MEAN_THRESH", "0.45"))
VIDEO_PEAK_THRESH = float(os.getenv("VIDEO_PEAK_THRESH", "0.60"))
VIDEO_PEAK_PCT    = int(os.getenv("VIDEO_PEAK_PCT", "95"))

# ----------- VITALS MODEL -----------
def load_vitals_model(model_dir="models"):
    path = os.path.join(model_dir, "vitals_baseline.joblib")
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
        return bundle  # {"scaler": ..., "logit": ...}
    except Exception as e:
        print(f"Failed to load vitals model: {e}")
        return None

def vitals_predict_proba(bundle, X):
    if bundle is None:
        # simple rule-based probability (0.15 normal, 0.65 elevated, 0.95 critical)
        probs = []
        for hr, temp, hrm, tm in X:
            score = 0
            if hr >= 120: score += 1
            if temp >= 37.8: score += 1
            probs.append(0.95 if score>=2 else (0.65 if score==1 else 0.15))
        return np.array(probs)
    scaler = bundle["scaler"]
    model = bundle["logit"]
    Xs = scaler.transform(X)
    return model.predict_proba(Xs)[:,1]

# ----------- VIDEO MODEL & FEATURES -----------
def simple_frame_features_from_dir(frames_dir: str):
    feats = []
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    for path in files:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, th = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)  # dark object on light bg
        M = cv2.moments(th)
        area = th.sum()/255.0
        if M["m00"] != 0:
            cx = M["m10"]/M["m00"]
            cy = M["m01"]/M["m00"]
        else:
            cx, cy = 0, 0
        feats.append([cx, cy, area])
    # Convert to sliding windows (win=8)
    X = []
    win = 8
    for i in range(len(feats)-win):
        window = np.array(feats[i:i+win]).flatten()
        X.append(window)
    X = np.array(X)
    return X, len(files)

def extract_frames_from_video(video_path: str, out_dir: str, fps: int = 10, size=(224,224)):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open uploaded video")
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(input_fps / fps)))
    idx, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            if size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, f"frame_{saved:05d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved

def load_video_model(model_dir="models"):
    path = os.path.join(model_dir, "video_baseline_rf.joblib")
    if not os.path.exists(path):
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        print(f"Failed to load video model: {e}")
        return None

def video_predict(
    video_path: str,
    model_dir: str = "models",
    mean_thresh: Optional[float] = None,
    peak_thresh: Optional[float] = None,
    peak_pct: Optional[int] = None,
) -> Dict[str, Any]:
    tmp = tempfile.mkdtemp(prefix="frames_")
    try:
        extract_frames_from_video(video_path, tmp, fps=10, size=(224,224))
        X, n_frames = simple_frame_features_from_dir(tmp)
        result = {"frames_processed": int(n_frames), "windows": int(len(X))}

        if len(X) == 0:
            result.update({"label":"normal","score":0.0,"detail":"Too few frames for analysis"})
            return result

        mt = VIDEO_MEAN_THRESH if mean_thresh is None else float(mean_thresh)
        pt = VIDEO_PEAK_THRESH if peak_thresh is None else float(peak_thresh)
        pp = VIDEO_PEAK_PCT if peak_pct is None else int(peak_pct)

        model = load_video_model(model_dir)
        if model is not None:
            probs = model.predict_proba(X)[:,1]
            mean_p = float(np.mean(probs))
            pxx    = float(np.percentile(probs, pp))
            label  = "fall" if (mean_p >= mt or pxx >= pt) else "normal"
            result.update({
                "label": label,
                "score": mean_p,
                "detail": f"Model RF mean + p{pp} rule",
                "stats": {"mean": mean_p, f"p{pp}": pxx},
                "params": {"mean_thresh": mt, "peak_thresh": pt, "peak_pct": pp}
            })
            return result

        # --- heuristic fallback ---
        cy_vals = []
        files = sorted(glob.glob(os.path.join(tmp, "frame_*.jpg")))
        for p in files:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            _, th = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
            M = cv2.moments(th)
            cy_vals.append((M["m01"]/M["m00"]) if M["m00"] != 0 else 0.0)

        if len(cy_vals) >= 10:
            q = len(cy_vals)//4
            delta = float(np.mean(cy_vals[-q:]) - np.mean(cy_vals[:q]))
        else:
            delta = 0.0

        score = float(min(0.99, max(0.01, 0.5 + delta/200)))
        label = "fall" if score >= mt else "normal"
        result.update({
            "label": label, "score": score, "detail": "Heuristic (no model found)",
            "stats": {"mean": score},
            "params": {"mean_thresh": mt, "peak_thresh": pt, "peak_pct": pp}
        })
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
