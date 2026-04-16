# Crowd Flow Analysis Tool

Real-time person detection, tracking, and movement analysis.

## Quick Start

### Phase 1: Video Processing & Detection

```bash
cd /Users/abhisekh/Desktop/supervision
python3 detect_people.py --model yolov8s.pt --conf 0.20 --imgsz 960 --flow-interval 5
```

Pick a video. Processing starts automatically.

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

## Output

### Phase 1 Output
```
output/phase1_processed/video_name/
├── video_name_result.mp4       # Annotated video
├── zone_time_series.csv        # Zone metrics over time
└── tracking_data.csv           # Frame-by-frame tracking
```

### Phase 2 Output
```
output/phase2_training/
├── phase2_lstm/
│   ├── phase2_lstm_checkpoint.pt     # Trained model
│   ├── phase2_lstm_predictions.csv   # LSTM predictions
│   ├── phase2_lstm_metrics.json      # Performance metrics
│   └── review_report/                # Detailed analysis
├── phase2_baseline/
│   ├── phase2_model_coefficients.csv # Ridge regression weights
│   └── phase2_predictions.csv        # Ridge predictions
├── METRICS_SUMMARY.md                # Performance comparison
└── phase2_model_comparison.md        # Model analysis
```

## Features

### Phase 1: Detection & Tracking
✓ Person detection (YOLOv8)
✓ ID tracking (ByteTrack)
✓ Movement arrows with speed coloring
✓ Grid zones with congestion status
✓ CSV export for ML models

### Phase 2: Predictive Modeling
✓ LSTM neural network for temporal sequence prediction
✓ Ridge regression baseline model
✓ Multi-model comparison and evaluation
✓ 3-second future prediction (people count & congestion status)

## Model Performance

| Model | Accuracy | MAE | Status |
|-------|----------|-----|--------|
| **Ridge Regression** | 96.3% | 0.729 | 🥇 Recommended (Production) |
| **Naive Baseline** | 96.3% | 0.704 | 🥈 Good Alternative |
| **LSTM Neural Network** | 94.4% | 1.281 | 🥉 Good (Complex) |

**Evaluation:** Fair comparison across all 4 videos with stratified train/val/test splits

## Controls

- **P** - Pause/Resume
- **Q** - Quit

## CSV Data

```
frame,tracker_id,cx,cy,arrow_dx,arrow_dy,speed,direction_degrees,zone_number,congestion_status
```

Ready for machine learning!
