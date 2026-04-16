import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


STATUS_TO_INT = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}

DIRECTION_TO_VECTOR = {
    "STILL": (0.0, 0.0),
    "E": (1.0, 0.0),
    "NE": (0.7071, -0.7071),
    "N": (0.0, -1.0),
    "NW": (-0.7071, -0.7071),
    "W": (-1.0, 0.0),
    "SW": (-0.7071, 0.7071),
    "S": (0.0, 1.0),
    "SE": (0.7071, 0.7071),
}

NUMERIC_FIELDS = [
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
]


@dataclass
class ZoneRow:
    video_name: str
    window_start_sec: int
    zone_number: int
    avg_people_count: float
    max_people_count: float
    unique_track_ids: float
    avg_speed_px_per_sec: float
    mean_flow_dx_per_frame: float
    mean_flow_dy_per_frame: float
    mean_optical_flow_dx_per_frame: float
    mean_optical_flow_dy_per_frame: float
    inflow_count: float
    outflow_count: float
    dominant_direction: str
    congestion_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 baseline: predict future zone crowd count/congestion."
    )
    parser.add_argument(
        "--input-root",
        default="output/phase1_processed",
        help="Phase 1 root directory, or one specific processed video folder.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=3,
        help="Number of past zone windows used as input features.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="How many future zone windows ahead to predict.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of each zone time-series used for training.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="L2 regularization strength for the linear baseline.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/phase2_training/phase2_baseline",
        help="Directory where Phase 2 datasets, predictions, and metrics are stored.",
    )
    return parser.parse_args()


def load_zone_rows(input_root: Path) -> list[ZoneRow]:
    zone_paths: list[Path]
    if (input_root / "zone_time_series.csv").exists():
        zone_paths = [input_root / "zone_time_series.csv"]
    else:
        zone_paths = sorted(input_root.glob("*/zone_time_series.csv"))
    if not zone_paths:
        raise FileNotFoundError(
            f"No zone_time_series.csv files found under {input_root.resolve()}"
        )

    rows: list[ZoneRow] = []
    for zone_path in zone_paths:
        video_name = zone_path.parent.name
        with open(zone_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    ZoneRow(
                        video_name=video_name,
                        window_start_sec=int(row["window_start_sec"]),
                        zone_number=int(row["zone_number"]),
                        avg_people_count=float(row["avg_people_count"]),
                        max_people_count=float(row["max_people_count"]),
                        unique_track_ids=float(row["unique_track_ids"]),
                        avg_speed_px_per_sec=float(row["avg_speed_px_per_sec"]),
                        mean_flow_dx_per_frame=float(row["mean_flow_dx_per_frame"]),
                        mean_flow_dy_per_frame=float(row["mean_flow_dy_per_frame"]),
                        mean_optical_flow_dx_per_frame=float(
                            row["mean_optical_flow_dx_per_frame"]
                        ),
                        mean_optical_flow_dy_per_frame=float(
                            row["mean_optical_flow_dy_per_frame"]
                        ),
                        inflow_count=float(row["inflow_count"]),
                        outflow_count=float(row["outflow_count"]),
                        dominant_direction=row["dominant_direction"],
                        congestion_status=row["congestion_status"],
                    )
                )
    return rows


def count_to_status(count: float) -> str:
    if count < 5:
        return "LOW"
    if count < 15:
        return "MEDIUM"
    if count < 30:
        return "HIGH"
    return "CRITICAL"


def encode_row_features(row: ZoneRow) -> dict[str, float]:
    direction_x, direction_y = DIRECTION_TO_VECTOR.get(
        row.dominant_direction, (0.0, 0.0)
    )
    return {
        "avg_people_count": row.avg_people_count,
        "max_people_count": row.max_people_count,
        "unique_track_ids": row.unique_track_ids,
        "avg_speed_px_per_sec": row.avg_speed_px_per_sec,
        "mean_flow_dx_per_frame": row.mean_flow_dx_per_frame,
        "mean_flow_dy_per_frame": row.mean_flow_dy_per_frame,
        "mean_optical_flow_dx_per_frame": row.mean_optical_flow_dx_per_frame,
        "mean_optical_flow_dy_per_frame": row.mean_optical_flow_dy_per_frame,
        "inflow_count": row.inflow_count,
        "outflow_count": row.outflow_count,
        "direction_x": direction_x,
        "direction_y": direction_y,
        "congestion_level": float(STATUS_TO_INT[row.congestion_status]),
    }


def build_samples(
    rows: list[ZoneRow], lookback: int, horizon: int
) -> tuple[list[str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[ZoneRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.video_name, row.zone_number)].append(row)

    feature_names = ["zone_number"]
    for lag in range(lookback):
        suffix = f"t_minus_{lag}"
        for base_name in NUMERIC_FIELDS + ["direction_x", "direction_y", "congestion_level"]:
            feature_names.append(f"{base_name}_{suffix}")
    if lookback >= 2:
        feature_names.extend(
            [
                "delta_avg_people_count",
                "delta_inflow_count",
                "delta_outflow_count",
                "delta_avg_speed_px_per_sec",
            ]
        )

    samples: list[dict[str, Any]] = []
    for (video_name, zone_number), sequence in grouped.items():
        ordered = sorted(sequence, key=lambda row: row.window_start_sec)
        if len(ordered) < lookback + horizon + 1:
            continue

        for current_index in range(lookback - 1, len(ordered) - horizon):
            history = ordered[current_index - lookback + 1 : current_index + 1]
            future = ordered[current_index + horizon]
            current = ordered[current_index]

            if future.window_start_sec - current.window_start_sec != horizon:
                continue

            sample: dict[str, Any] = {
                "video_name": video_name,
                "zone_number": zone_number,
                "current_window_start_sec": current.window_start_sec,
                "target_window_start_sec": future.window_start_sec,
                "target_avg_people_count": future.avg_people_count,
                "target_congestion_status": future.congestion_status,
                "current_avg_people_count": current.avg_people_count,
                "current_congestion_status": current.congestion_status,
            }

            sample["zone_number"] = float(zone_number)
            encoded_history = [encode_row_features(row) for row in reversed(history)]
            for lag, encoded in enumerate(encoded_history):
                suffix = f"t_minus_{lag}"
                for key, value in encoded.items():
                    sample[f"{key}_{suffix}"] = float(value)

            if lookback >= 2:
                current_features = encoded_history[0]
                previous_features = encoded_history[1]
                sample["delta_avg_people_count"] = (
                    current_features["avg_people_count"]
                    - previous_features["avg_people_count"]
                )
                sample["delta_inflow_count"] = (
                    current_features["inflow_count"] - previous_features["inflow_count"]
                )
                sample["delta_outflow_count"] = (
                    current_features["outflow_count"] - previous_features["outflow_count"]
                )
                sample["delta_avg_speed_px_per_sec"] = (
                    current_features["avg_speed_px_per_sec"]
                    - previous_features["avg_speed_px_per_sec"]
                )

            samples.append(sample)

    return feature_names, samples


def split_samples(
    samples: list[dict[str, Any]], train_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[(sample["video_name"], int(sample["zone_number"]))].append(sample)

    train_samples: list[dict[str, Any]] = []
    test_samples: list[dict[str, Any]] = []

    for key in sorted(grouped):
        ordered = sorted(
            grouped[key], key=lambda sample: int(sample["current_window_start_sec"])
        )
        if len(ordered) < 3:
            train_samples.extend(ordered)
            continue

        split_index = max(1, min(len(ordered) - 1, int(len(ordered) * train_ratio)))
        train_samples.extend(ordered[:split_index])
        test_samples.extend(ordered[split_index:])

    return train_samples, test_samples


def samples_to_matrix(
    samples: list[dict[str, Any]], feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [[float(sample[name]) for name in feature_names] for sample in samples],
        dtype=np.float64,
    )
    y = np.array(
        [float(sample["target_avg_people_count"]) for sample in samples],
        dtype=np.float64,
    )
    return x, y


def fit_ridge_regression(
    x: np.ndarray, y: np.ndarray, ridge_alpha: float
) -> np.ndarray:
    x_with_bias = np.column_stack([np.ones(len(x)), x])
    regularizer = np.eye(x_with_bias.shape[1], dtype=np.float64) * ridge_alpha
    regularizer[0, 0] = 0.0
    return np.linalg.solve(
        x_with_bias.T @ x_with_bias + regularizer,
        x_with_bias.T @ y,
    )


def predict_with_coefficients(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    x_with_bias = np.column_stack([np.ones(len(x)), x])
    predictions = x_with_bias @ coefficients
    return np.clip(predictions, a_min=0.0, a_max=None)


def classification_accuracy(actual: list[str], predicted: list[str]) -> float:
    if not actual:
        return 0.0
    matches = sum(1 for a, b in zip(actual, predicted) if a == b)
    return matches / len(actual)


def build_confusion(actual: list[str], predicted: list[str]) -> dict[str, dict[str, int]]:
    labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    matrix = {label: {inner: 0 for inner in labels} for label in labels}
    for actual_label, predicted_label in zip(actual, predicted):
        matrix.setdefault(actual_label, {inner: 0 for inner in labels})
        matrix[actual_label].setdefault(predicted_label, 0)
        matrix[actual_label][predicted_label] += 1
    return matrix


def evaluate_predictions(
    samples: list[dict[str, Any]], predicted_counts: np.ndarray
) -> dict[str, Any]:
    actual_counts = np.array(
        [float(sample["target_avg_people_count"]) for sample in samples], dtype=np.float64
    )
    actual_statuses = [str(sample["target_congestion_status"]) for sample in samples]
    predicted_statuses = [count_to_status(value) for value in predicted_counts]

    mae = float(np.mean(np.abs(predicted_counts - actual_counts))) if len(samples) else 0.0
    rmse = (
        float(np.sqrt(np.mean((predicted_counts - actual_counts) ** 2)))
        if len(samples)
        else 0.0
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "congestion_accuracy": classification_accuracy(
            actual_statuses, predicted_statuses
        ),
        "confusion_matrix": build_confusion(actual_statuses, predicted_statuses),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)

    rows = load_zone_rows(input_root=input_root)
    feature_names, samples = build_samples(
        rows=rows,
        lookback=max(1, args.lookback),
        horizon=max(1, args.horizon),
    )
    if not samples:
        raise ValueError("Not enough zone time-series data to build Phase 2 samples.")

    train_samples, test_samples = split_samples(
        samples=samples, train_ratio=args.train_ratio
    )
    if not test_samples:
        raise ValueError(
            "No test samples were created. Add more videos or reduce the train ratio."
        )

    train_x, train_y = samples_to_matrix(train_samples, feature_names)
    test_x, _test_y = samples_to_matrix(test_samples, feature_names)

    coefficients = fit_ridge_regression(
        x=train_x, y=train_y, ridge_alpha=float(args.ridge_alpha)
    )
    test_predictions = predict_with_coefficients(test_x, coefficients)

    naive_predictions = np.array(
        [float(sample["current_avg_people_count"]) for sample in test_samples],
        dtype=np.float64,
    )

    model_metrics = evaluate_predictions(test_samples, test_predictions)
    naive_metrics = evaluate_predictions(test_samples, naive_predictions)

    supervised_rows: list[dict[str, Any]] = []
    for sample in train_samples:
        row = {name: sample[name] for name in feature_names}
        row.update(
            {
                "split": "train",
                "video_name": sample["video_name"],
                "current_window_start_sec": sample["current_window_start_sec"],
                "target_window_start_sec": sample["target_window_start_sec"],
                "target_avg_people_count": sample["target_avg_people_count"],
                "target_congestion_status": sample["target_congestion_status"],
            }
        )
        supervised_rows.append(row)
    for sample in test_samples:
        row = {name: sample[name] for name in feature_names}
        row.update(
            {
                "split": "test",
                "video_name": sample["video_name"],
                "current_window_start_sec": sample["current_window_start_sec"],
                "target_window_start_sec": sample["target_window_start_sec"],
                "target_avg_people_count": sample["target_avg_people_count"],
                "target_congestion_status": sample["target_congestion_status"],
            }
        )
        supervised_rows.append(row)

    prediction_rows: list[dict[str, Any]] = []
    for sample, predicted_count, naive_count in zip(
        test_samples, test_predictions, naive_predictions
    ):
        prediction_rows.append(
            {
                "video_name": sample["video_name"],
                "zone_number": sample["zone_number"],
                "current_window_start_sec": sample["current_window_start_sec"],
                "target_window_start_sec": sample["target_window_start_sec"],
                "current_avg_people_count": round(
                    float(sample["current_avg_people_count"]), 3
                ),
                "actual_future_avg_people_count": round(
                    float(sample["target_avg_people_count"]), 3
                ),
                "predicted_future_avg_people_count": round(float(predicted_count), 3),
                "naive_future_avg_people_count": round(float(naive_count), 3),
                "actual_future_congestion_status": sample["target_congestion_status"],
                "predicted_future_congestion_status": count_to_status(
                    float(predicted_count)
                ),
                "naive_future_congestion_status": count_to_status(float(naive_count)),
            }
        )

    coefficient_rows = [
        {"feature": "bias", "coefficient": round(float(coefficients[0]), 6)}
    ] + [
        {"feature": name, "coefficient": round(float(value), 6)}
        for name, value in zip(feature_names, coefficients[1:])
    ]

    metrics = {
        "config": {
            "input_root": str(input_root),
            "lookback": int(args.lookback),
            "horizon": int(args.horizon),
            "train_ratio": float(args.train_ratio),
            "ridge_alpha": float(args.ridge_alpha),
        },
        "dataset": {
            "zone_rows": len(rows),
            "supervised_samples": len(samples),
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "videos_used": sorted({row.video_name for row in rows}),
        },
        "model_metrics": model_metrics,
        "naive_baseline_metrics": naive_metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "phase2_supervised_dataset.csv",
        [
            "split",
            "video_name",
            "current_window_start_sec",
            "target_window_start_sec",
            *feature_names,
            "target_avg_people_count",
            "target_congestion_status",
        ],
        supervised_rows,
    )
    write_csv(
        output_dir / "phase2_predictions.csv",
        [
            "video_name",
            "zone_number",
            "current_window_start_sec",
            "target_window_start_sec",
            "current_avg_people_count",
            "actual_future_avg_people_count",
            "predicted_future_avg_people_count",
            "naive_future_avg_people_count",
            "actual_future_congestion_status",
            "predicted_future_congestion_status",
            "naive_future_congestion_status",
        ],
        prediction_rows,
    )
    write_csv(
        output_dir / "phase2_model_coefficients.csv",
        ["feature", "coefficient"],
        coefficient_rows,
    )
    with open(output_dir / "phase2_metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Phase 2 supervised dataset: {output_dir / 'phase2_supervised_dataset.csv'}")
    print(f"Phase 2 predictions: {output_dir / 'phase2_predictions.csv'}")
    print(f"Phase 2 coefficients: {output_dir / 'phase2_model_coefficients.csv'}")
    print(f"Phase 2 metrics: {output_dir / 'phase2_metrics.json'}")
    print(
        "Model test MAE: "
        f"{metrics['model_metrics']['mae']:.3f} | "
        "Model congestion accuracy: "
        f"{metrics['model_metrics']['congestion_accuracy']:.3f}"
    )
    print(
        "Naive baseline MAE: "
        f"{metrics['naive_baseline_metrics']['mae']:.3f} | "
        "Naive congestion accuracy: "
        f"{metrics['naive_baseline_metrics']['congestion_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
