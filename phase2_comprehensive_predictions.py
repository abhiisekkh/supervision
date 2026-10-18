#!/usr/bin/env python3
"""
Generate comprehensive predictions for ALL videos in phase1_processed.
Shows predictions vs actuals for every zone frame with detailed metrics.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


STATUS_TO_INT = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}

INT_TO_STATUS = {v: k for k, v in STATUS_TO_INT.items()}

FEATURE_NAMES = [
    "avg_people_count",
    "max_people_count",
    "unique_track_ids",
    "avg_speed_px_per_sec",
    "mean_flow_dx_per_frame",
    "mean_flow_dy_per_frame",
    "mean_optical_flow_dx_per_frame",
    "mean_optical_flow_dy_per_frame",
    "inflow_count",
    "outflow_count",
    "direction_x",
    "direction_y",
    "congestion_level",
]


@dataclass
class ZoneRow:
    video_name: str
    window_start_sec: int
    zone_number: int
    values: dict[str, float]


def count_to_status(count: float) -> str:
    count = max(0.0, float(count))
    if count < 5:
        return "LOW"
    if count < 15:
        return "MEDIUM"
    if count < 30:
        return "HIGH"
    return "CRITICAL"


def load_zone_rows(input_root: Path) -> list[ZoneRow]:
    zone_paths = sorted(input_root.glob("*/zone_time_series.csv"))
    if not zone_paths:
        raise FileNotFoundError(f"No zone_time_series.csv files found under {input_root}")

    rows: list[ZoneRow] = []
    for zone_path in zone_paths:
        video_name = zone_path.parent.name
        with open(zone_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                direction_x, direction_y = (0.0, 0.0)  # Simplified for comprehensive predictions
                rows.append(
                    ZoneRow(
                        video_name=video_name,
                        window_start_sec=int(row["window_start_sec"]),
                        zone_number=int(row["zone_number"]),
                        values={
                            "avg_people_count": float(row["avg_people_count"]),
                            "max_people_count": float(row["max_people_count"]),
                            "unique_track_ids": float(row["unique_track_ids"]),
                            "avg_speed_px_per_sec": float(row["avg_speed_px_per_sec"]),
                            "mean_flow_dx_per_frame": float(row["mean_flow_dx_per_frame"]),
                            "mean_flow_dy_per_frame": float(row["mean_flow_dy_per_frame"]),
                            "mean_optical_flow_dx_per_frame": float(row["mean_optical_flow_dx_per_frame"]),
                            "mean_optical_flow_dy_per_frame": float(row["mean_optical_flow_dy_per_frame"]),
                            "inflow_count": float(row["inflow_count"]),
                            "outflow_count": float(row["outflow_count"]),
                            "direction_x": direction_x,
                            "direction_y": direction_y,
                            "congestion_level": float(STATUS_TO_INT.get(row.get("congestion_status", "LOW"), 0)),
                        },
                    )
                )
    return rows


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray, float, float]:
    """Load LSTM checkpoint and normalization stats."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_mean = np.array(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.array(checkpoint["feature_std"], dtype=np.float32)
    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])
    return checkpoint, feature_mean, feature_std, target_mean, target_std


class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, prediction_horizon: int):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(x)
        decoder_input = x[:, -1:, 0:1]
        outputs = []
        for _ in range(self.prediction_horizon):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            step_prediction = self.output(decoder_output)
            outputs.append(step_prediction)
            decoder_input = step_prediction
        return torch.cat(outputs, dim=1).squeeze(-1)


def make_predictions_for_all_zones(
    rows: list[ZoneRow],
    model: EncoderDecoderLSTM,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: float,
    target_std: float,
    lookback_seconds: int,
    predict_seconds: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Generate predictions for all zones in all videos."""
    
    # Group by (video, zone)
    grouped: dict[tuple[str, int], list[ZoneRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.video_name, row.zone_number)].append(row)

    predictions_list: list[dict[str, Any]] = []
    model.eval()

    with torch.no_grad():
        for (video_name, zone_number), sequence in grouped.items():
            ordered = sorted(sequence, key=lambda row: row.window_start_sec)

            # Generate predictions for all valid windows
            for start_index in range(len(ordered) - lookback_seconds - predict_seconds + 1):
                history = ordered[start_index : start_index + lookback_seconds]
                future = ordered[start_index + lookback_seconds : start_index + lookback_seconds + predict_seconds]

                if len(future) != predict_seconds:
                    continue

                # Check for continuous timestamps
                expected_seconds = list(
                    range(history[0].window_start_sec, future[-1].window_start_sec + 1)
                )
                actual_seconds = [row.window_start_sec for row in history + future]
                if actual_seconds != expected_seconds:
                    continue

                # Build feature matrix
                input_features = np.array(
                    [[row.values[name] for name in FEATURE_NAMES] for row in history],
                    dtype=np.float32,
                )
                target_counts = np.array(
                    [row.values["avg_people_count"] for row in future], dtype=np.float32
                )

                # Normalize
                x_normalized = (input_features - feature_mean) / feature_std
                x_tensor = torch.tensor(x_normalized, dtype=torch.float32).unsqueeze(0).to(device)

                # Predict
                y_pred_normalized = model(x_tensor).cpu().numpy()[0]
                y_pred = y_pred_normalized * target_std + target_mean
                y_pred = np.clip(y_pred, 0, None)

                # Save predictions for each forecast step
                for step_index, (pred, actual) in enumerate(zip(y_pred, target_counts), start=1):
                    target_second = future[step_index - 1].window_start_sec
                    predictions_list.append(
                        {
                            "video_name": video_name,
                            "zone_number": zone_number,
                            "input_start_sec": history[0].window_start_sec,
                            "input_end_sec": history[-1].window_start_sec,
                            "target_second": target_second,
                            "forecast_step": step_index,
                            "actual_people_count": round(float(actual), 2),
                            "predicted_people_count": round(float(pred), 2),
                            "error": round(abs(float(pred) - float(actual)), 2),
                            "percent_error": round(abs(float(pred) - float(actual)) / max(float(actual), 1.0) * 100, 1),
                            "actual_status": count_to_status(float(actual)),
                            "predicted_status": count_to_status(float(pred)),
                            "status_correct": 1 if count_to_status(float(actual)) == count_to_status(float(pred)) else 0,
                        }
                    )

    return predictions_list


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive predictions for all zones.")
    parser.add_argument("--input-root", default="output/phase1_processed", help="Phase 1 root directory")
    parser.add_argument("--checkpoint", default="output/phase2_training/phase2_lstm_multidata_retrained/phase2_lstm_checkpoint.pt", help="LSTM checkpoint path")
    parser.add_argument("--output-dir", default="output/phase2_training/phase2_comprehensive_predictions", help="Output directory")
    parser.add_argument("--lookback-seconds", type=int, default=8)
    parser.add_argument("--predict-seconds", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print("📊 Loading zone data from all videos...")
    rows = load_zone_rows(input_root)
    print(f"   Loaded {len(rows)} zone rows from {len({r.video_name for r in rows})} videos")

    print("🧠 Loading LSTM checkpoint...")
    checkpoint, feature_mean, feature_std, target_mean, target_std = load_checkpoint(checkpoint_path, device)
    
    config = checkpoint.get("config", {})
    hidden_size = config.get("hidden_size", 64)
    num_layers = config.get("num_layers", 2)

    model = EncoderDecoderLSTM(
        input_size=len(FEATURE_NAMES),
        hidden_size=hidden_size,
        num_layers=num_layers,
        prediction_horizon=args.predict_seconds,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("🎯 Generating predictions for all zones...")
    predictions = make_predictions_for_all_zones(
        rows=rows,
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        lookback_seconds=args.lookback_seconds,
        predict_seconds=args.predict_seconds,
        device=device,
    )

    print(f"✅ Generated {len(predictions)} predictions")

    # Write detailed predictions
    predictions_path = output_dir / "comprehensive_predictions.csv"
    write_csv(
        predictions_path,
        [
            "video_name",
            "zone_number",
            "input_start_sec",
            "input_end_sec",
            "target_second",
            "forecast_step",
            "actual_people_count",
            "predicted_people_count",
            "error",
            "percent_error",
            "actual_status",
            "predicted_status",
            "status_correct",
        ],
        predictions,
    )
    print(f"📈 Predictions saved to {predictions_path}")

    # Generate summary statistics
    if predictions:
        predictions_array = np.array([p["predicted_people_count"] for p in predictions])
        actuals_array = np.array([p["actual_people_count"] for p in predictions])
        errors = np.abs(predictions_array - actuals_array)
        
        status_correct = sum(p["status_correct"] for p in predictions)
        status_accuracy = status_correct / len(predictions) * 100

        summary = {
            "total_predictions": len(predictions),
            "videos": sorted({p["video_name"] for p in predictions}),
            "num_videos": len({p["video_name"] for p in predictions}),
            "zones_per_video": sorted({p["zone_number"] for p in predictions}),
            "num_zones": len({(p["video_name"], p["zone_number"]) for p in predictions}),
            "mae": float(np.mean(errors)),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mean_percent_error": float(np.mean([p["percent_error"] for p in predictions])),
            "status_accuracy_percent": round(status_accuracy, 2),
            "normalization": {
                "feature_mean": feature_mean.tolist(),
                "feature_std": feature_std.tolist(),
                "target_mean": target_mean,
                "target_std": target_std,
            },
        }

        metrics_path = output_dir / "comprehensive_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📊 Summary metrics saved to {metrics_path}")

        print(f"\n✨ COMPREHENSIVE PREDICTIONS SUMMARY:")
        print(f"   Videos: {summary['num_videos']} ({', '.join(summary['videos'])})")
        print(f"   Zones: {summary['num_zones']} (zone numbers: {summary['zones_per_video']})")
        print(f"   Total predictions: {summary['total_predictions']}")
        print(f"   MAE: {summary['mae']:.2f} people")
        print(f"   RMSE: {summary['rmse']:.2f} people")
        print(f"   Mean % Error: {summary['mean_percent_error']:.1f}%")
        print(f"   Status Accuracy: {summary['status_accuracy_percent']:.1f}%")


if __name__ == "__main__":
    main()
