# Convenience targets for local development
.PHONY: setup vitals frames train api app docker

setup:
\tpython -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

vitals:
\tpython src/simulate_vitals.py --out data/vitals/vitals_train.csv --minutes 120 --seed 42

frames:
\tpython src/preprocess_video.py --video data/raw/toy/toy_fall_sim.mp4 --out data/raw/toy/frames

train:
\tpython src/train_baselines.py --vitals data/vitals/vitals_train.csv --frames_dir data/raw/toy/frames --model_out models

api:
\tuvicorn src.api:app --reload --port 8000

app:
\tstreamlit run streamlit_app/app.py

docker:
\tdocker compose up --build
