# Predictive Modeling for Crowd Flow Analysis and Congestion Detection in Large Public Gatherings

An end-to-end computer vision and spatio-temporal deep learning platform designed to extract crowd flow vectors, track individual movement trajectories, and predict localized congestion spikes ahead of time.

---

## Quick Start

### Installation & Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/abhiisekkh/supervision.git
cd supervision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Phase 1: Video Processing & Detection

```bash
cd /Users/abhisekh/Desktop/supervision
python3 detect_people.py --model yolov8s.pt --conf 0.20 --imgsz 960 --flow-interval 5
```

Pick a video. Processing starts automatically.

---

### Phase 2: LSTM Model Training

```bash
cd /Users/abhisekh/Desktop/supervision && python3 phase2_lstm_train.py \
  --input-root output/phase1_processed \
  --output-dir output/phase2_training/phase2_lstm \
  --lookback-seconds 8 \
  --predict-seconds 3 \
  --epochs 40 \
  --batch-size 16 \
  --hidden-size 64 \
  --num-layers 2 \
  --device cpu
```

---

### Interactive Web Platform (Streamlit)

Launch the full interactive operations platform:

```bash
streamlit run app.py
```

Open **`http://localhost:8501`** in your browser.

#### Platform Modules:
1. **Project Overview & Home**: Architecture diagram, metrics summary, and interactive user guide.
2. **Step 1: Run Detection & Live Preview**: Video playback (Spacebar to play/pause, $\pm5s$ seeking), drag-and-drop video upload, parameter configuration, and live pipeline execution.
3. **Step 2: Inspect Saved Processed Videos**: Processed video playback with detection and flow overlays, plus a **Folder Artifacts table where every CSV can be opened directly in a new browser tab**.
4. **Step 3: Compare Model Predictions vs Actual**: Side-by-side ground truth vs. predicted crowd count, categorical congestion status, automated MATCH/MISMATCH verdicts, and correlation scatter plots.
5. **Step 4: Full CSV Data & Metrics Inspection**: Complete table viewer for `zone_time_series.csv`, Phase 2 metrics JSON, and comprehensive predictions.
6. **Step 5: Model Architecture Comparison & Benchmark**: Empirical comparison across 6 models (Attention-LSTM, Super Ensemble, Ridge Regression, Gradient Boosting, KNN, and Naive Baseline).

---

## Directory Structure & Outputs

```
supervision/
├── app.py                            # Streamlit interactive web platform
├── detect_people.py                  # Phase 1: YOLOv8 + ByteTrack + Lucas-Kanade optical flow
├── phase2_lstm_train.py              # Phase 2: Attention-LSTM training pipeline
├── phase2_lstm_infer.py              # Phase 2: Sequential inference and future forecasting
├── phase2_baseline.py                # Ridge, Gradient Boosting, KNN, and Ensemble baselines
├── sample_data/                      # Self-contained review video & sample datasets for cloud deployment
│   ├── 6387-191695740.mp4            # Default review dataset video
│   ├── phase1_processed/             # Sample processed CSVs and preview MP4
│   └── phase2_training/              # Sample model predictions
├── requirements.txt                  # Python dependencies
├── station_config.json               # Spatial calibration parameters
├── output/
│   ├── phase1_processed/             # Extracted flow vectors, tracklets, and zone time series
│   │   └── <video_name>/
│   │       ├── <video_name>_result.mp4
│   │       ├── zone_time_series.csv
│   │       ├── tracked_detections.csv
│   │       └── flow_vectors.csv
│   └── phase2_training/              # Trained models, metrics, and multi-step predictions
└── README.md
```

---

## Model Benchmark & Evaluation Summary

Evaluated on multi-source public gathering datasets:

| Model Architecture | Model Family | Test MAE ↓ | Test RMSE ↓ | Accuracy ↑ | Inference | Status |
|---|---|---|---|---|---|---|
| **Attention-LSTM Neural Net** | Deep Recurrent + Attention | **0.3027** | **0.5393** | **98.94%** | 0.80 ms | **Recommended (Best Flow Learner)** |
| **Super Ensemble** | Stacked (Ridge+GB+KNN) | 0.3796 | 0.6400 | 94.85% | 0.45 ms | Competitive Baseline |
| **Ridge Regression** | Regularized L2 Linear | 0.4842 | 0.8143 | 98.04% | 0.08 ms | Fast Edge Baseline |
| **Gradient Boosting** | Ensemble Decision Trees | 0.3899 | 0.6732 | 94.04% | 0.32 ms | Non-Linear Baseline |
| **K-Nearest Neighbors** | Instance Based (k=5) | 0.5265 | 0.8892 | 93.22% | 0.60 ms | Local Density Reference |
| **Naive Persistence** | Zero-Learning ($t \to t+1$) | 0.4417 | 0.7871 | 98.04% | < 0.01 ms | Lagging Reference |

---

## Live Video Controls (In Web Platform & Native OpenCV)

- **Spacebar**: Play / Pause video preview
- **Arrow Keys (← / →)**: Seek backwards / forward $\pm5$ seconds
- **Q**: Quit OpenCV preview window
