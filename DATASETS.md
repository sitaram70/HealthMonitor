# DATASETS

> **Important:** Review each dataset’s license/ethics notes and obtain any required access before use. Do not redistribute datasets unless the license explicitly allows it.

## Fall / Activity Video Datasets
- **UR Fall Detection (URFD)** — falls vs. activities of daily living in indoor scenes; short clips; ideal for CPU baselines.
- **SisFall** — simulated falls and ADLs from multiple subjects; widely used for fall detection.
- **UP-Fall** — multimodal (video + wearables like IMU/ECG). Great if you want to fuse signals later.

**How to fetch**  
See `src/fetch_urfd.py` for helper URLs. Example:
```bash
python src/fetch_urfd.py --dataset urfd --dest data/raw/urfd
```
Unpack archives into `data/raw/<dataset>` and document your selected subset and splits.

## Physiological / Vitals Time-Series
- **PhysioNet / MIT-BIH** waveform databases (e.g., Normal Sinus Rhythm, selected MIMIC waveform subsets) provide ECG/HR signals you can downsample to heart-rate series.
- For classroom speed, start with the **simulated vitals** generator (`src/simulate_vitals.py`) and later replace with real series.

## Classroom recommendations
- Start with the built-in **toy video** + **simulated vitals** to validate the pipeline.
- Add a **small, balanced subset** (e.g., 20–40 short clips) from URFD/SisFall/UP-Fall for training/evaluation.
- Keep a **data sheet** per team: sources, consent, preprocessing, splits, class balance, known biases, and limitations.
