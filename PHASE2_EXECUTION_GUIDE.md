# Phase 2 Execution Guide & Quick Start

## Overview
Phase 2 implements three different approaches to predict future crowd congestion states:
1. **LSTM Temporal Neural Network** - Deep learning model for sequence prediction
2. **Ridge Regression** - Linear model with L2 regularization
3. **Naive Baseline** - Simple forward projection

---

## Prerequisites

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Required packages (already in pyproject.toml)
# - torch
# - numpy
# - scipy
# - csv, json (stdlib)
```

### Input Data
Phase 2 requires Phase 1 output:
```
output/phase1_processed/
├── <video1>/zone_time_series.csv
├── <video2>/zone_time_series.csv
└── <video3>/zone_time_series.csv
```

---

## Models & Scripts

### 1. LSTM Temporal Model (`phase2_lstm_train.py`)

**Purpose**: Train LSTM neural network for sequence-to-sequence prediction

**Command**:
```bash
python3 phase2_lstm_train.py \
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

**Parameters**:
- `--input-root`: Path to Phase 1 processed data
- `--lookback-seconds`: Historical context (8 seconds = 8 windows)
- `--predict-seconds`: Future horizon (3 seconds = 3 windows)
- `--epochs`: Training iterations (40 is good for 36 sequences)
- `--batch-size`: Batch size (16 is standard)
- `--hidden-size`: LSTM hidden dimension (64 is balanced)
- `--num-layers`: Stacked LSTM layers (2 recommended)
- `--device`: GPU or CPU

**Output**:
```
output/phase2_training/phase2_lstm/
├── phase2_lstm_checkpoint.pt        # Trained model weights
├── phase2_lstm_metrics.json         # Performance metrics
├── phase2_lstm_history.csv          # Training history
├── phase2_lstm_predictions.csv      # Model predictions
└── review_report/                   # Analysis reports
    ├── phase2_review_summary.md
    ├── phase2_confusion_matrix.csv
    ├── phase2_step_metrics.csv
    └── phase2_zone_summary.csv
```

**Expected Results**:
- Test MAE: ~1.07 people
- Congestion Accuracy: ~92.6%
- Training Time: ~2 minutes (CPU)

---

### 2. LSTM Inference (`phase2_lstm_infer.py`)

**Purpose**: Generate predictions using trained LSTM model

**Command**:
```bash
python3 phase2_lstm_infer.py \
  --input-root output/phase1_processed \
  --checkpoint output/phase2_training/phase2_lstm/phase2_lstm_checkpoint.pt \
  --output-dir output/phase2_training/phase2_lstm_inference
```

**Output**:
```
output/phase2_training/phase2_lstm_inference/
├── phase2_lstm_inference_metrics.json
└── phase2_lstm_inference_predictions.csv
```

---

### 3. Ridge Regression Baseline (`phase2_baseline.py`)

**Purpose**: Train linear ridge regression model for comparison

**Command**:
```bash
python3 phase2_baseline.py \
  --input-root output/phase1_processed \
  --lookback 3 \
  --horizon 1 \
  --train-ratio 0.8 \
  --ridge-alpha 1.0 \
  --output-dir output/phase2_training/phase2_baseline
```

**Parameters**:
- `--lookback`: Number of past windows (3 = 3×5sec = 15 sec history)
- `--horizon`: Prediction steps ahead (1 = 5 seconds ahead)
- `--train-ratio`: Train/test split (0.8 = 80/20)
- `--ridge-alpha`: L2 regularization strength (1.0 is good default)

**Output**:
```
output/phase2_training/phase2_baseline/
├── phase2_metrics.json             # Performance metrics
├── phase2_supervised_dataset.csv   # Train/test data
├── phase2_predictions.csv          # Predictions
└── phase2_model_coefficients.csv   # Feature weights
```

**Expected Results**:
- Test MAE: ~0.73 people
- Congestion Accuracy: ~96.3%
- Training Time: <1 second

---

### 4. Review Report Generation (`phase2_review_report.py`)

**Purpose**: Generate detailed analysis of LSTM predictions

**Command**:
```bash
python3 phase2_review_report.py \
  --input-root output/phase1_processed \
  --predictions output/phase2_training/phase2_lstm/phase2_lstm_predictions.csv \
  --output-dir output/phase2_training/phase2_lstm/review_report
```

**Output**: Confusion matrices, step metrics, zone summaries

---

### 5. Model Comparison (`phase2_model_comparison.py`)

**Purpose**: Compare all three models side-by-side

**Command**:
```bash
python3 phase2_model_comparison.py
```

**Output**:
```
output/phase2_training/phase2_model_comparison.md
```

**Shows**:
- MAE comparison (LSTM vs Ridge vs Naive)
- Accuracy comparison
- Recommendation for production use

---

### 6. Metrics Summary (`phase2_metrics_summary.py`)

**Purpose**: Create visual performance summary

**Command**:
```bash
python3 phase2_metrics_summary.py
```

**Output**:
```
output/phase2_training/METRICS_SUMMARY.md
```

**Shows**:
- ASCII art performance charts
- Leaderboard (✅ Ridge Regression wins)
- Detailed breakdown of each model

---

## Full Phase 2 Pipeline (Run All at Once)

Execute this to run complete Phase 2 from scratch:

```bash
#!/bin/bash
set -e

echo "=== Phase 2: Complete Pipeline ==="

echo "1. Training LSTM..."
python3 phase2_lstm_train.py \
  --input-root output/phase1_processed \
  --output-dir output/phase2_training/phase2_lstm \
  --lookback-seconds 8 --predict-seconds 3 \
  --epochs 40 --batch-size 16 --hidden-size 64 --num-layers 2 --device cpu

echo "2. LSTM Inference..."
python3 phase2_lstm_infer.py \
  --input-root output/phase1_processed \
  --checkpoint output/phase2_training/phase2_lstm/phase2_lstm_checkpoint.pt \
  --output-dir output/phase2_training/phase2_lstm_inference

echo "3. LSTM Review Report..."
python3 phase2_review_report.py \
  --input-root output/phase1_processed \
  --predictions output/phase2_training/phase2_lstm/phase2_lstm_predictions.csv \
  --output-dir output/phase2_training/phase2_lstm/review_report

echo "4. Ridge Regression Baseline..."
python3 phase2_baseline.py \
  --input-root output/phase1_processed \
  --lookback 3 --horizon 1 --train-ratio 0.8 --ridge-alpha 1.0 \
  --output-dir output/phase2_training/phase2_baseline

echo "5. Model Comparison..."
python3 phase2_model_comparison.py

echo "6. Metrics Summary..."
python3 phase2_metrics_summary.py

echo "=== Phase 2 Complete ==="
echo "Review outputs at:"
echo "  - output/phase2_training/phase2_model_comparison.md"
echo "  - output/phase2_training/METRICS_SUMMARY.md"
echo "  - PHASE2_REVIEW.md"
```

---

## Output Files Reference

### LSTM Outputs
| File | Purpose | Size |
|------|---------|------|
| phase2_lstm_checkpoint.pt | Trained weights | ~150KB |
| phase2_lstm_metrics.json | Config & metrics | ~5KB |
| phase2_lstm_history.csv | Epoch-by-epoch loss | ~1KB |
| phase2_lstm_predictions.csv | Train/val/test predictions | ~3KB |
| review_report/*.csv | Analysis tables | ~10KB |
| review_report/*.md | Detailed report | ~5KB |

### Ridge Regression Outputs
| File | Purpose | Size |
|------|---------|------|
| phase2_metrics.json | Results & data info | ~4KB |
| phase2_supervised_dataset.csv | All train/test data | ~20KB |
| phase2_predictions.csv | Predictions | ~2KB |
| phase2_model_coefficients.csv | Feature weights | ~1KB |

### Summary Documents
| File | Purpose |
|------|---------|
| phase2_model_comparison.md | Detailed model comparison |
| METRICS_SUMMARY.md | Visual performance charts |
| PHASE2_REVIEW.md | Comprehensive review document |

---

## Key Metrics Reference

### Model Performance

| Model | MAE | RMSE | Accuracy |
|-------|-----|------|----------|
| **Ridge Regression** ⭐ | **0.729** | **0.916** | **96.3%** |
| Naive Baseline | **0.704** | 1.245 | 96.3% |
| LSTM | 1.071 | 1.342 | 92.6% |

### Interpretation
1. **Ridge Regression**: Best balance of accuracy and interpretability
2. **Naive Baseline**: Shows auto-correlation is strong in 3-sec window
3. **LSTM**: Over-fits, but useful for longer horizons

---

## Troubleshooting

### LSTM Training Slow?
- Increase batch size (32, 64) - trades memory for speed
- Use GPU with `--device cuda` (requires NVIDIA GPU + CUDA)
- Reduce epochs to 20

### Out of Memory?
- Reduce batch size to 8
- Reduce hidden size to 32
- Reduce sequence length (lookback-seconds)

### Models Not Generating Outputs?
- Verify Phase 1 data exists in `output/phase1_processed/`
- Check file permissions
- Verify CSV headers match expected names

### Metrics Look Wrong?
- Verify train/test split is consistent
- Check for NaN values in Phase 1 data
- Review input CSV files for data quality

---

## Next Steps

### After Phase 2 Review
1. ✅ Compare all three models
2. ✅ Select production model (Ridge Regression recommended)
3. ✅ Review metrics and recommendations
4. ⏭️ **Phase 3**: Production deployment

### Phase 3 Planning
- REST API for real-time predictions
- Dashboard visualization
- Live video stream testing
- Docker containerization

---

## File Structure Overview

```
supervision/
├── detect_people.py                    # Phase 1 (video analysis)
├── phase2_lstm_train.py               # Phase 2 (LSTM training)
├── phase2_lstm_infer.py               # Phase 2 (LSTM inference)
├── phase2_review_report.py            # Phase 2 (LSTM analysis)
├── phase2_baseline.py                 # Phase 2 (Ridge regression)
├── phase2_model_comparison.py         # Phase 2 (Compare models)
├── phase2_metrics_summary.py          # Phase 2 (Summary viz)
├── PHASE2_REVIEW.md                   # Phase 2 (Main review)
│
└── output/
    ├── phase1_processed/              # Phase 1 outputs
    │   ├── video1/zone_time_series.csv
    │   ├── video2/zone_time_series.csv
    │   └── video3/zone_time_series.csv
    │
    └── phase2_training/               # Phase 2 outputs
        ├── phase2_lstm/
        │   ├── checkpoint.pt
        │   ├── metrics.json
        │   ├── predictions.csv
        │   └── review_report/
        │
        ├── phase2_baseline/
        │   ├── metrics.json
        │   ├── predictions.csv
        │   └── coefficients.csv
        │
        ├── phase2_lstm_inference/
        │
        ├── phase2_model_comparison.md
        ├── METRICS_SUMMARY.md
        └── phase2_review_report.md
```

---

## Quick Start (Already Done)

All Phase 2 models have been successfully trained! You can review the results:

```bash
# View LSTM metrics
cat output/phase2_training/phase2_lstm/phase2_lstm_metrics.json | jq

# View Ridge metrics
cat output/phase2_training/phase2_baseline/phase2_metrics.json | jq

# View comparison
less output/phase2_training/phase2_model_comparison.md

# View summary
less output/phase2_training/METRICS_SUMMARY.md

# View full review
less PHASE2_REVIEW.md
```

---

## Getting Help

For each script, use:
```bash
python3 <script>.py --help
```

This shows all available parameters and defaults.

---

**Phase 2 Status**: ✅ Complete & Ready for Review
**All outputs generated**: ✅ Yes
**Ready for Phase 3**: ✅ Yes

*Last Updated: April 15, 2026*
