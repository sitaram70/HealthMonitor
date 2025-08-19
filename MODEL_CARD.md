# MODEL CARD — Fall Detector (Random-Forest on windowed features)

**Intended Use**  
Educational demo for short **fall/normal** classification on CPU laptops. **Not for clinical use.**

**Inputs & Preprocessing**  
- Short video clip (≈10–30 s).  
- Frames sampled at **10 FPS**, resized **224×224**.  
- Cheap per-frame features from binary mask moments (centroid x/y, area), aggregated over sliding windows.

**Model**  
- `RandomForestClassifier` on window features.  
- Decision uses an **event-aware rule**:
```
label = 'fall' if mean(prob) ≥ mean_thresh OR p(peak_pct) ≥ peak_thresh
```
Defaults (UI-tunable): `mean_thresh=0.45`, `peak_thresh=0.60`, `peak_pct=95`.

**Training Data**  
- Toy clip for sanity; small balanced subset from public fall datasets (URFD/SisFall/UP-Fall).  
- Document exact sources/splits in project reports.

**Metrics**  
- Report Accuracy, Precision/Recall/F1, ROC-AUC on held-out clips.  
- Include confusion matrix and per-class metrics.  
- Measure **latency** (end-to-end) and memory on CPU hardware used in class.

**Limitations**  
- Simple features; sensitive to lighting/background and camera motion.  
- Multiple people or occlusions not handled.  
- Mean probability can dilute short events; mitigated via the **peak (pXX)** rule.  
- **NOT a medical device.**

**Ethical Considerations**  
- Use consented/anonymized data only; avoid personally identifying content.  
- Provide a *misuse statement* (no surveillance/disciplinary use).  
- Document biases (subject age, clothing, lighting) and known failure modes.
