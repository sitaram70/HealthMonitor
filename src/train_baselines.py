#!/usr/bin/env python3
import argparse, os, glob, cv2, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def simple_frame_features(frames_dir):
    # very cheap features: per-frame centroid & area of the darkest object (toy video compatible)
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
    # convert to sliding windows (label synthetic "fall" near the end half)
    X, y = [], []
    win = 8
    for i in range(len(feats)-win):
        window = np.array(feats[i:i+win]).flatten()
        # synthetic label: frames in later half are more likely "fall"
        label = 1 if i > (len(feats)//2) else 0
        X.append(window); y.append(label)
    return np.array(X), np.array(y)

def vitals_features(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    # engineer rolling stats
    df["hr_roll_mean"] = df["heart_rate_bpm"].rolling(15, min_periods=1).mean()
    df["temp_roll_mean"] = df["temp_c"].rolling(30, min_periods=1).mean()
    # target = risk_event (already simulated)
    y = df["risk_event"].values
    X = df[["heart_rate_bpm","temp_c","hr_roll_mean","temp_roll_mean"]].fillna(method="bfill").values
    # reduce to per-5s samples
    step = 5
    X = X[::step]; y = y[::step]
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vitals", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--model_out", default="models")
    args = ap.parse_args()
    os.makedirs(args.model_out, exist_ok=True)

    # Video features
    Xv, yv = simple_frame_features(args.frames_dir)
    Xv_tr, Xv_te, yv_tr, yv_te = train_test_split(Xv, yv, test_size=0.25, random_state=42, stratify=yv)

    # Vitals features
    Xs, ys = vitals_features(args.vitals)
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=0.25, random_state=42, stratify=ys)

    # Standardize vitals
    scaler = StandardScaler().fit(Xs_tr)
    Xs_tr_s = scaler.transform(Xs_tr)
    Xs_te_s = scaler.transform(Xs_te)

    # Train simple baselines
    logit = LogisticRegression(max_iter=200).fit(Xs_tr_s, ys_tr)
    rf = RandomForestClassifier(n_estimators=150, random_state=42).fit(Xv_tr, yv_tr)

    # Eval
    vitals_proba = logit.predict_proba(Xs_te_s)[:,1]
    vitals_auc = roc_auc_score(ys_te, vitals_proba)
    print(f"[Vitals] AUROC: {vitals_auc:.3f}")
    print(classification_report(ys_te, (vitals_proba>0.5).astype(int)))

    video_proba = rf.predict_proba(Xv_te)[:,1]
    video_auc = roc_auc_score(yv_te, video_proba)
    print(f"[Video] AUROC: {video_auc:.3f}")
    print(classification_report(yv_te, (video_proba>0.5).astype(int)))

    # Save models
    import joblib
    joblib.dump({"scaler": scaler, "logit": logit}, os.path.join(args.model_out, "vitals_baseline.joblib"))
    joblib.dump(rf, os.path.join(args.model_out, "video_baseline_rf.joblib"))
    print(f"Saved models to {args.model_out}")

if __name__ == "__main__":
    main()
