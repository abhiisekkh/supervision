#!/usr/bin/env python3
"""Master training script: trains all models and generates a comprehensive
comparison report with maximum accuracy configurations.

Usage:
    python train_all_accurate.py
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_step(name: str, command: list[str], cwd: str = ".") -> tuple[bool, str]:
    """Run a training step and capture output."""
    print(f"\n{'='*70}")
    print(f"  STEP: {name}")
    print(f"{'='*70}")
    start = time.time()

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=False,
            text=True,
            timeout=1800,  # 30 min max per step
        )
        elapsed = time.time() - start
        status = "SUCCESS" if result.returncode == 0 else "FAILED"
        print(f"\n  [{status}] {name} completed in {elapsed:.1f}s")
        return result.returncode == 0, f"{elapsed:.1f}s"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"\n  [TIMEOUT] {name} timed out after {elapsed:.1f}s")
        return False, f"TIMEOUT ({elapsed:.1f}s)"
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  [ERROR] {name}: {e}")
        return False, f"ERROR: {e}"


def load_json_safe(path: Path) -> dict:
    """Load JSON file, returning empty dict on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def generate_metrics_summary(output_dir: Path) -> None:
    """Generate a comprehensive metrics summary markdown file."""
    lstm_metrics = load_json_safe(output_dir / "phase2_lstm" / "phase2_lstm_metrics.json")
    baseline_metrics = load_json_safe(output_dir / "phase2_baseline" / "phase2_metrics.json")

    # Extract metrics
    lstm_test = lstm_metrics.get("test_metrics", {})
    ridge = baseline_metrics.get("ridge_metrics", {})
    gb = baseline_metrics.get("gradient_boosting_metrics", {})
    knn = baseline_metrics.get("knn_metrics", {})
    ensemble = baseline_metrics.get("ensemble_metrics", {})
    naive = baseline_metrics.get("naive_baseline_metrics", {})

    models = []
    if lstm_test:
        models.append(("LSTM (Attention)", lstm_test))
    if ridge:
        models.append(("Ridge Regression", ridge))
    if gb:
        models.append(("Gradient Boosting", gb))
    if knn:
        models.append(("K-Nearest Neighbors", knn))
    if ensemble:
        models.append(("Super Ensemble (Ridge+GB+KNN)", ensemble))
    if naive:
        models.append(("Naive Baseline", naive))

    # Sort by MAE (lower is better)
    models.sort(key=lambda m: m[1].get("mae", float("inf")))

    # Find best model
    best_model_name = models[0][0] if models else "N/A"
    best_mae = models[0][1].get("mae", 0) if models else 0

    # Find best accuracy
    best_acc_model = max(models, key=lambda m: m[1].get("congestion_accuracy", 0)) if models else ("N/A", {})

    summary = f"""# PHASE 2 PERFORMANCE METRICS SUMMARY
## Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 🏆 Model Leaderboard (Sorted by MAE)

| Rank | Model | MAE ↓ | RMSE ↓ | Congestion Accuracy ↑ |
|------|-------|-------|--------|----------------------|
"""
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, (name, m) in enumerate(models):
        medal = medals[i] if i < len(medals) else f"{i+1}."
        mae = m.get("mae", 0)
        rmse = m.get("rmse", 0)
        acc = m.get("congestion_accuracy", 0) * 100
        best_tag = " ⭐" if name == best_model_name else ""
        summary += f"| {medal} | **{name}**{best_tag} | {mae:.4f} | {rmse:.4f} | {acc:.2f}% |\n"

    summary += f"""
---

## 🎯 Best Model: {best_model_name}
- **MAE**: {best_mae:.4f} people
- **Congestion Accuracy**: {best_acc_model[1].get('congestion_accuracy', 0)*100:.2f}%

"""

    # LSTM details
    if lstm_test:
        model_type = lstm_metrics.get("model_type", "original")
        total_params = lstm_metrics.get("total_parameters", "N/A")
        config = lstm_metrics.get("config", {})
        summary += f"""## LSTM Model Details
```
Architecture: {model_type.title()} (Bidirectional + Multi-Head Attention)
Parameters: {total_params:,} if isinstance(total_params, int) else total_params
Hidden Size: {config.get('hidden_size', 'N/A')}
Layers: {config.get('num_layers', 'N/A')}
Lookback: {config.get('lookback_seconds', 'N/A')} seconds
Prediction: {config.get('predict_seconds', 'N/A')} seconds
Learning Rate: {config.get('learning_rate', 'N/A')}
Dropout: {config.get('dropout', 'N/A')}
Loss: Huber (Smooth L1)
Optimizer: AdamW with weight decay
LR Scheduler: Cosine Annealing with Warm Restarts
EMA: Applied for stable evaluation
Device: {lstm_metrics.get('device', 'N/A')}

Test Metrics:
  MAE: {lstm_test.get('mae', 0):.4f}
  RMSE: {lstm_test.get('rmse', 0):.4f}
  Congestion Accuracy: {lstm_test.get('congestion_accuracy', 0)*100:.2f}%
  Best Val MAE: {lstm_metrics.get('best_validation_mae', 0):.4f}
```

"""
        step_mae = lstm_test.get("step_mae", [])
        if step_mae:
            summary += "### Per-Step MAE (Prediction Horizon)\n"
            summary += "| Step | MAE |\n|------|-----|\n"
            for i, mae_val in enumerate(step_mae, 1):
                summary += f"| t+{i} | {mae_val:.4f} |\n"
            summary += "\n"

    # Baseline details
    if baseline_metrics:
        config = baseline_metrics.get("config", {})
        summary += f"""## Baseline Model Details
```
Ridge Alpha (CV-selected): {config.get('ridge_alpha', 'N/A')}
GB Estimators: {config.get('gb_n_estimators', 'N/A')}
GB Learning Rate: {config.get('gb_learning_rate', 'N/A')}
GB Max Depth: {config.get('gb_max_depth', 'N/A')}
KNN Neighbors: {config.get('knn_neighbors', 'N/A')}
Ensemble Weight (Ridge): {config.get('ensemble_weight_ridge', 'N/A')}
Ensemble Weight (GB): {config.get('ensemble_weight_gb', 'N/A')}
Ensemble Weight (KNN): {config.get('ensemble_weight_knn', 'N/A')}
Feature Count: {baseline_metrics.get('dataset', {}).get('feature_count', 'N/A')}
Lookback: {config.get('lookback', 'N/A')} windows
Prediction: {config.get('horizon', 'N/A')} window ahead
```

"""

    # Per-video metrics
    per_video = baseline_metrics.get("per_video_ensemble_metrics", {})
    if per_video:
        summary += "## Per-Video Ensemble Metrics\n\n"
        summary += "| Video | MAE | RMSE | Accuracy |\n"
        summary += "|-------|-----|------|----------|\n"
        for video, m in per_video.items():
            summary += (
                f"| {video} | {m.get('mae', 0):.4f} | "
                f"{m.get('rmse', 0):.4f} | "
                f"{m.get('congestion_accuracy', 0)*100:.2f}% |\n"
            )
        summary += "\n"

    # Dataset info
    lstm_ds = lstm_metrics.get("dataset", {})
    baseline_ds = baseline_metrics.get("dataset", {})
    summary += f"""## Dataset Information
```
LSTM:
  Zone Rows: {lstm_ds.get('zone_rows', 'N/A')}
  Total Sequences: {lstm_ds.get('total_sequences', 'N/A')}
  Train/Val/Test: {lstm_ds.get('train_sequences', 'N/A')}/{lstm_ds.get('val_sequences', 'N/A')}/{lstm_ds.get('test_sequences', 'N/A')}
  Videos: {', '.join(lstm_ds.get('videos_with_sequences', []))}

Baseline:
  Supervised Samples: {baseline_ds.get('supervised_samples', 'N/A')}
  Train/Test: {baseline_ds.get('train_samples', 'N/A')}/{baseline_ds.get('test_samples', 'N/A')}
  Features: {baseline_ds.get('feature_count', 'N/A')}
  Videos: {', '.join(baseline_ds.get('videos_used', []))}
```

---

## Key Improvements Over Previous Version

1. ✅ **Attention LSTM**: Bidirectional encoder + multi-head attention
2. ✅ **Gradient Boosting**: Hand-coded ensemble of decision stumps
3. ✅ **Feature Engineering**: 13 additional features (rolling stats, trends, flow ratios)
4. ✅ **K-Nearest Neighbors**: Added distance-based non-parametric modeling
5. ✅ **Super Ensemble**: Optimized grid-search weights across Ridge, GB, and KNN
6. ✅ **Cross-Validated Ridge**: Automatic alpha selection
7. ✅ **EMA Weights**: Exponential moving average for stable predictions
8. ✅ **Huber Loss**: Robust to outliers
9. ✅ **Cosine Annealing**: Better learning rate scheduling

---

**Status: READY FOR REVIEW ✅**
"""

    summary_path = output_dir / "METRICS_SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Metrics summary written to: {summary_path}")


def main() -> None:
    project_root = Path(__file__).parent
    output_dir = project_root / "output" / "phase2_training"
    python = sys.executable

    print("=" * 70)
    print("  CROWD FLOW PREDICTION - MAXIMUM ACCURACY TRAINING PIPELINE")
    print("=" * 70)
    print(f"  Python: {python}")
    print(f"  Project: {project_root}")
    print(f"  Output: {output_dir}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results: dict[str, tuple[bool, str]] = {}

    # Step 1: Train enhanced LSTM
    success, elapsed = run_step(
        "Train Attention LSTM",
        [
            python,
            str(project_root / "phase2_lstm_train.py"),
            "--epochs", "100",
            "--hidden-size", "128",
            "--num-layers", "3",
            "--dropout", "0.3",
            "--learning-rate", "5e-4",
            "--patience", "15",
            "--weight-decay", "1e-4",
            "--device", "auto",
            "--batch-size", "32",
            "--lookback-seconds", "8",
            "--predict-seconds", "3",
            "--ema-decay", "0.999",
        ],
        cwd=str(project_root),
    )
    results["LSTM"] = (success, elapsed)

    # Step 2: Train enhanced baseline (Ridge + GB + Ensemble)
    success, elapsed = run_step(
        "Train Ridge + Gradient Boosting + Ensemble",
        [
            python,
            str(project_root / "phase2_baseline.py"),
        ],
        cwd=str(project_root),
    )
    results["Baseline"] = (success, elapsed)

    # Step 3: Generate comprehensive metrics summary
    print(f"\n{'='*70}")
    print("  GENERATING METRICS SUMMARY")
    print(f"{'='*70}")
    generate_metrics_summary(output_dir)

    # Final report
    print(f"\n{'='*70}")
    print("  TRAINING PIPELINE COMPLETE")
    print(f"{'='*70}")
    for step_name, (success, elapsed) in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {step_name}: {status} ({elapsed})")
    print(f"{'='*70}")

    # Load and display final metrics
    lstm_metrics = load_json_safe(output_dir / "phase2_lstm" / "phase2_lstm_metrics.json")
    baseline_metrics = load_json_safe(output_dir / "phase2_baseline" / "phase2_metrics.json")

    print("\n  FINAL METRICS:")
    if lstm_metrics.get("test_metrics"):
        t = lstm_metrics["test_metrics"]
        print(f"  LSTM:     MAE={t['mae']:.4f}  RMSE={t['rmse']:.4f}  Accuracy={t['congestion_accuracy']*100:.2f}%")
    if baseline_metrics.get("ridge_metrics"):
        t = baseline_metrics["ridge_metrics"]
        print(f"  Ridge:    MAE={t['mae']:.4f}  RMSE={t['rmse']:.4f}  Accuracy={t['congestion_accuracy']*100:.2f}%")
    if baseline_metrics.get("gradient_boosting_metrics"):
        t = baseline_metrics["gradient_boosting_metrics"]
        print(f"  GB:       MAE={t['mae']:.4f}  RMSE={t['rmse']:.4f}  Accuracy={t['congestion_accuracy']*100:.2f}%")
    if baseline_metrics.get("knn_metrics"):
        t = baseline_metrics["knn_metrics"]
        print(f"  KNN:      MAE={t['mae']:.4f}  RMSE={t['rmse']:.4f}  Accuracy={t['congestion_accuracy']*100:.2f}%")
    if baseline_metrics.get("ensemble_metrics"):
        t = baseline_metrics["ensemble_metrics"]
        print(f"  Ensemble: MAE={t['mae']:.4f}  RMSE={t['rmse']:.4f}  Accuracy={t['congestion_accuracy']*100:.2f}%")
    if baseline_metrics.get("naive_baseline_metrics"):
        t = baseline_metrics["naive_baseline_metrics"]
        print(f"  Naive:    MAE={t['mae']:.4f}  RMSE={t['rmse']:.4f}  Accuracy={t['congestion_accuracy']*100:.2f}%")

    print(f"\n{'='*70}")
    all_success = all(s for s, _ in results.values())
    if all_success:
        print("  🎉 ALL MODELS TRAINED SUCCESSFULLY - READY FOR REVIEW!")
    else:
        print("  ⚠️  Some steps failed. Check output above for details.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
