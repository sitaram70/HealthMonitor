#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from datetime import datetime, timedelta
rng = np.random.default_rng()

def simulate(minutes=120, seed=42, anomaly_prob=0.02):
    rng = np.random.default_rng(seed)
    start = datetime.now().replace(second=0, microsecond=0)
    times = [start + timedelta(seconds=i) for i in range(minutes*60)]
    hr = []
    temp = []
    risk = []
    base_hr = rng.normal(78, 5)
    base_temp = rng.normal(36.8, 0.15)
    for t in range(len(times)):
        # Random walk with small noise
        if t % 30 == 0:
            base_hr += rng.normal(0, 0.5)
        if t % 120 == 0:
            base_temp += rng.normal(0, 0.02)
        hr_val = base_hr + rng.normal(0, 1.2)
        temp_val = base_temp + rng.normal(0, 0.05)
        # occasional anomalies (spikes)
        if rng.random() < anomaly_prob:
            hr_val += rng.uniform(20, 40)  # HR spike
            temp_val += rng.uniform(0.6, 1.2)  # feverish spike
            risk.append(1)
        else:
            risk.append(0)
        hr.append(max(45, min(180, hr_val)))
        temp.append(max(35.5, min(40.5, temp_val)))
    df = pd.DataFrame({
        "timestamp": times,
        "heart_rate_bpm": np.round(hr, 1),
        "temp_c": np.round(temp, 2),
        "risk_event": risk
    })
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/vitals/vitals_train.csv")
    ap.add_argument("--minutes", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--anomaly_prob", type=float, default=0.02)
    args = ap.parse_args()
    df = simulate(minutes=args.minutes, seed=args.seed, anomaly_prob=args.anomaly_prob)
    out = args.out
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df)} rows.")
if __name__ == "__main__":
    main()
