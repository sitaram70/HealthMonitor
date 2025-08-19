# AI-Powered Health Monitoring (CPU-only)

**Streamlit + FastAPI** app for vitals monitoring and fall detection. Designed to run on laptops with **CPU only** (no GPU required).

---

## Features
- **Vitals Stream (CSV/API)**: real-time charts, rule-based risk, alert table.
- **Video Test (Upload)**: upload a short clip → classify **fall/normal** using a Random-Forest on sliding-window features.
- **Event-aware decision rule**: FALL if `mean(prob) ≥ mean_thresh` **OR** `p(peak_pct) ≥ peak_thresh`.
- **Tunable thresholds** from the UI (no restart) or via environment variables.
- **Dockerized** (UI + API) or pure Python virtualenv.
- **Education-ready**: simple baselines, reproducible, CPU-friendly.

---

## Architecture
- **Streamlit UI** (`streamlit_app/app.py`)
  - *Vitals Stream* tab: KPI tiles, rolling charts, alerts table.
  - *Video Test (Upload)* tab: file uploader, threshold controls, results panel.
- **FastAPI Backend** (`src/api.py`)
  - `/predict/video` — classify a clip; returns label, score, stats, and params used.
  - `/stream/*` — simple vitals stream helpers for the dashboard.
- **Inference** (`src/inference.py`)
  - Loads models from `models/` if present; otherwise uses safe heuristics.
  - Implements the **mean + pXX** event-aware rule.

> _Tip:_ See `src/train_baselines.py` for CPU-friendly training pipelines and `src/preprocess_video.py` for frame extraction.

---

## Repository layout (typical)
```
.
├── data/
│   ├── raw/
│   │   ├── toy/                  # toy_fall_sim.mp4 + extracted frames
│   │   └── urfd/                 # (optional) download datasets here
│   └── vitals/                   # vitals CSVs (simulated or real)
├── docs/
│   └── img/                      # screenshots for README
├── models/                       # saved .joblib files
├── notebooks/                    # EDA + training walkthroughs
├── src/
│   ├── api.py                    # FastAPI server
│   ├── inference.py              # model loading + prediction logic
│   ├── preprocess_video.py       # frame extraction (CPU-friendly)
│   ├── simulate_vitals.py        # HR/Temp simulator
│   ├── train_baselines.py        # vitals + video baselines
│   └── fetch_urfd.py             # dataset fetch helper (URLs + licensing notes)
├── streamlit_app/
│   └── app.py                    # dashboard
├── Dockerfile.api
├── Dockerfile.app
├── docker-compose.yml
├── Makefile
├── README.md
├── DATASETS.md
├── MODEL_CARD.md
└── requirements.txt
```

---

## Quickstart (venv)
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

# Generate vitals & frames (CPU-only)
python src/simulate_vitals.py --out data/vitals/vitals_train.csv --minutes 120 --seed 42
python src/preprocess_video.py --video data/raw/toy/toy_fall_sim.mp4 --out data/raw/toy/frames

# Train simple baselines (saves models/*.joblib)
python src/train_baselines.py --vitals data/vitals/vitals_train.csv --frames_dir data/raw/toy/frames --model_out models

# Run services
uvicorn src.api:app --reload --port 8000
streamlit run streamlit_app/app.py
```

Open the UI → **API Mode** (`http://localhost:8000`) → **Video Test (Upload)** → upload a 10–30s clip.

---

## Thresholds (UI or env)
You can tune thresholds from the **Video Test (Upload)** tab — no restart required.  
Alternatively, set environment variables before starting the API:

```bash
export VIDEO_MEAN_THRESH=0.45
export VIDEO_PEAK_THRESH=0.60
export VIDEO_PEAK_PCT=95
```

The API echoes the parameters actually used in the JSON response under `"params"` so students can record settings.

---

## Docker (optional)
```bash
docker compose up --build
# UI:  http://localhost:8501
# API: http://localhost:8000
```

---

## API examples
**Classify a clip with custom thresholds:**
```bash
curl -s -X POST "http://localhost:8000/predict/video"   -F file=@data/raw/toy/toy_fall_sim.mp4   -F mean_thresh=0.45 -F peak_thresh=0.60 -F peak_pct=95 | jq
```
**Health check:**
```bash
curl http://localhost:8000/health
```

---

## Datasets & ethics
See **DATASETS.md** for recommended sources (URFD, SisFall, UP-Fall, PhysioNet) and licensing notes.  
Always document consent, anonymize where possible, and include a short **data sheet** with sources, preprocessing, splits, biases, and limitations.

---

## Teaching alignment
This repo maps to a standard 16-week capstone:
- **Milestone 1**: Proposal + system diagram.
- **Milestone 2**: Data/EDA + vitals simulator + sample video frames.
- **Milestone 3**: Baseline models + metrics.
- **Milestone 4**: Integrated MVP (Streamlit + FastAPI).
- **Milestone 5**: Final demo (video upload) + documentation.

---

## Screenshots
Place PNGs in `docs/img/` and reference here, e.g.:
```
![Vitals stream](docs/img/vitals_stream.png)
![Video test](docs/img/video_test.png)
```

---

## License
Choose a license (MIT is common for class projects). Create `LICENSE` at the repo root.
