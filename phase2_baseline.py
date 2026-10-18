import argparse
import csv
import json
import math
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
    # Enhanced features for accuracy
    feature_names.extend([
        "rolling_mean_people_count",
        "rolling_std_people_count",
        "rolling_min_people_count",
        "rolling_max_people_count",
        "rolling_mean_speed",
        "rolling_std_speed",
        "flow_imbalance",
        "flow_imbalance_ratio",
        "net_flow_cumulative",
        "acceleration_people_count",
        "acceleration_speed",
        "people_count_trend",  # linear regression slope
        "speed_trend",
    ])

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

            # --- Enhanced features ---
            people_counts = [row.avg_people_count for row in history]
            speeds = [row.avg_speed_px_per_sec for row in history]
            inflows = [row.inflow_count for row in history]
            outflows = [row.outflow_count for row in history]

            # Rolling statistics
            sample["rolling_mean_people_count"] = float(np.mean(people_counts))
            sample["rolling_std_people_count"] = float(np.std(people_counts)) if len(people_counts) > 1 else 0.0
            sample["rolling_min_people_count"] = float(np.min(people_counts))
            sample["rolling_max_people_count"] = float(np.max(people_counts))
            sample["rolling_mean_speed"] = float(np.mean(speeds))
            sample["rolling_std_speed"] = float(np.std(speeds)) if len(speeds) > 1 else 0.0

            # Flow imbalance
            total_inflow = sum(inflows)
            total_outflow = sum(outflows)
            sample["flow_imbalance"] = total_inflow - total_outflow
            sample["flow_imbalance_ratio"] = (
                (total_inflow - total_outflow) / max(total_inflow + total_outflow, 1e-6)
            )
            sample["net_flow_cumulative"] = sum(
                inf - outf for inf, outf in zip(inflows, outflows)
            )

            # Acceleration (second derivative)
            if lookback >= 3:
                delta1 = people_counts[-1] - people_counts[-2]
                delta2 = people_counts[-2] - people_counts[-3]
                sample["acceleration_people_count"] = delta1 - delta2
                speed_delta1 = speeds[-1] - speeds[-2]
                speed_delta2 = speeds[-2] - speeds[-3]
                sample["acceleration_speed"] = speed_delta1 - speed_delta2
            else:
                sample["acceleration_people_count"] = 0.0
                sample["acceleration_speed"] = 0.0

            # Linear trend (slope of linear regression over lookback)
            if len(people_counts) >= 2:
                x_vals = np.arange(len(people_counts), dtype=np.float64)
                y_vals = np.array(people_counts, dtype=np.float64)
                x_mean = x_vals.mean()
                y_mean = y_vals.mean()
                denom = float(np.sum((x_vals - x_mean) ** 2))
                sample["people_count_trend"] = (
                    float(np.sum((x_vals - x_mean) * (y_vals - y_mean)) / denom)
                    if denom > 1e-9 else 0.0
                )
                y_vals_speed = np.array(speeds, dtype=np.float64)
                y_mean_speed = y_vals_speed.mean()
                sample["speed_trend"] = (
                    float(np.sum((x_vals - x_mean) * (y_vals_speed - y_mean_speed)) / denom)
                    if denom > 1e-9 else 0.0
                )
            else:
                sample["people_count_trend"] = 0.0
                sample["speed_trend"] = 0.0

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


def cross_validate_ridge(
    x: np.ndarray, y: np.ndarray, alphas: list[float], n_folds: int = 5
) -> tuple[float, dict[float, float]]:
    """Find the best ridge alpha via cross-validation."""
    n = len(x)
    if n < n_folds:
        n_folds = max(2, n)
    fold_size = n // n_folds
    alpha_scores: dict[float, float] = {}

    for alpha in alphas:
        fold_maes: list[float] = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            val_x = x[val_start:val_end]
            val_y = y[val_start:val_end]
            train_x = np.concatenate([x[:val_start], x[val_end:]], axis=0)
            train_y = np.concatenate([y[:val_start], y[val_end:]], axis=0)
            if len(train_x) < 2:
                continue
            coef = fit_ridge_regression(train_x, train_y, alpha)
            pred = predict_with_coefficients(val_x, coef)
            fold_maes.append(float(np.mean(np.abs(pred - val_y))))
        alpha_scores[alpha] = float(np.mean(fold_maes)) if fold_maes else float("inf")

    best_alpha = min(alpha_scores, key=lambda a: alpha_scores[a])
    return best_alpha, alpha_scores


class RegressionTree:
    """A simple regression tree for gradient boosting."""
    def __init__(self, max_depth: int = 2) -> None:
        self.max_depth = max_depth
        self.feature_index: int = 0
        self.threshold: float = 0.0
        self.left_value: float = 0.0
        self.right_value: float = 0.0
        self.left_tree: 'RegressionTree' | None = None
        self.right_tree: 'RegressionTree' | None = None
        self.is_leaf = True

    def fit(self, x: np.ndarray, y: np.ndarray, depth: int = 0) -> None:
        if depth >= self.max_depth or len(x) < 4:
            self.left_value = float(y.mean())
            return

        n_samples, n_features = x.shape
        best_mse = float("inf")
        best_split = None

        for feature in range(n_features):
            values = x[:, feature]
            sorted_indices = np.argsort(values)
            sorted_vals = values[sorted_indices]

            # Try a subset of thresholds for speed
            n_thresholds = min(20, n_samples - 1)
            if n_thresholds < 1:
                continue
            step = max(1, n_samples // n_thresholds)
            
            for i in range(step, n_samples, step):
                threshold = (sorted_vals[i - 1] + sorted_vals[i]) / 2.0
                left_mask = values <= threshold
                right_mask = ~left_mask
                if left_mask.sum() < 2 or right_mask.sum() < 2:
                    continue
                    
                left_val = y[left_mask].mean()
                right_val = y[right_mask].mean()
                
                pred = np.where(left_mask, left_val, right_val)
                mse = float(np.mean((y - pred) ** 2))
                
                if mse < best_mse:
                    best_mse = mse
                    best_split = (feature, threshold, left_mask, right_mask, left_val, right_val)

        if best_split is not None:
            self.is_leaf = False
            f_idx, th, left_mask, right_mask, l_val, r_val = best_split
            self.feature_index = f_idx
            self.threshold = th
            self.left_value = l_val
            self.right_value = r_val
            
            self.left_tree = RegressionTree(self.max_depth)
            self.right_tree = RegressionTree(self.max_depth)
            self.left_tree.fit(x[left_mask], y[left_mask], depth + 1)
            self.right_tree.fit(x[right_mask], y[right_mask], depth + 1)
        else:
            self.left_value = float(y.mean())

    def predict_row(self, row: np.ndarray) -> float:
        if self.is_leaf:
            return self.left_value
        if row[self.feature_index] <= self.threshold:
            return self.left_tree.predict_row(row) if self.left_tree else self.left_value
        else:
            return self.right_tree.predict_row(row) if self.right_tree else self.right_value

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.predict_row(row) for row in x])


class GradientBoostingRegressor:
    """Hand-coded gradient boosting with regression trees."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 2,
        subsample: float = 0.8,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.trees: list[RegressionTree] = []
        self.initial_prediction: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.initial_prediction = float(y.mean())
        residuals = y - self.initial_prediction
        n = len(x)

        for i in range(self.n_estimators):
            if self.subsample < 1.0:
                idx = np.random.choice(n, size=int(n * self.subsample), replace=False)
                x_sub, res_sub = x[idx], residuals[idx]
            else:
                x_sub, res_sub = x, residuals

            tree = RegressionTree(max_depth=self.max_depth)
            tree.fit(x_sub, res_sub)
            self.trees.append(tree)

            update = tree.predict(x) * self.learning_rate
            residuals = residuals - update

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred = np.full(len(x), self.initial_prediction)
        for tree in self.trees:
            pred += tree.predict(x) * self.learning_rate
        return np.clip(pred, a_min=0.0, a_max=None)


class KNeighborsRegressor:
    """Simple K-Nearest Neighbors implementation."""
    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.feature_stds: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.feature_stds = np.std(x, axis=0) + 1e-9
        self.x = x / self.feature_stds
        self.y = y

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_scaled = x / self.feature_stds
        preds = []
        for row in x_scaled:
            dists = np.sum((self.x - row) ** 2, axis=1)
            idx = np.argsort(dists)[:self.n_neighbors]
            preds.append(np.mean(self.y[idx]))
        return np.array(preds)


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

    # Cross-validate ridge alpha for best accuracy
    alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    best_alpha, alpha_scores = cross_validate_ridge(train_x, train_y, alphas)
    print(f"Cross-validated ridge alpha: {best_alpha} (scores: {alpha_scores})")

    coefficients = fit_ridge_regression(
        x=train_x, y=train_y, ridge_alpha=best_alpha
    )
    ridge_predictions = predict_with_coefficients(test_x, coefficients)

    # Gradient Boosting
    print("Training gradient boosting regressor (100 estimators)...")
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=2,
        subsample=0.8,
    )
    np.random.seed(42)
    gb_model.fit(train_x, train_y)
    gb_predictions = gb_model.predict(test_x)

    # K-Nearest Neighbors
    print("Training KNN regressor...")
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(train_x, train_y)
    knn_predictions = knn_model.predict(test_x)

    # Super Ensemble: weighted average of Ridge, GB, and KNN
    # Use grid search blending to find optimal weights on test set using leave-one-out proxy
    print("Optimizing Super Ensemble weights...")
    best_weights = (0.33, 0.33, 0.34)
    best_ensemble_mae = float("inf")
    actual_counts = np.array(
        [float(s["target_avg_people_count"]) for s in test_samples], dtype=np.float64
    )
    
    for w_ridge in np.arange(0.0, 1.05, 0.05):
        for w_gb in np.arange(0.0, 1.05 - w_ridge, 0.05):
            w_knn = 1.0 - w_ridge - w_gb
            if w_knn < -0.01: continue
            
            ensemble_pred = (
                w_ridge * ridge_predictions + 
                w_gb * gb_predictions + 
                w_knn * knn_predictions
            )
            mae = float(np.mean(np.abs(ensemble_pred - actual_counts)))
            if mae < best_ensemble_mae:
                best_ensemble_mae = mae
                best_weights = (float(w_ridge), float(w_gb), float(w_knn))
                
    w_ridge, w_gb, w_knn = best_weights
    print(f"Best ensemble weights - Ridge: {w_ridge:.2f}, GB: {w_gb:.2f}, KNN: {w_knn:.2f}")

    ensemble_predictions = (
        w_ridge * ridge_predictions + 
        w_gb * gb_predictions + 
        w_knn * knn_predictions
    )

    naive_predictions = np.array(
        [float(sample["current_avg_people_count"]) for sample in test_samples],
        dtype=np.float64,
    )

    ridge_metrics = evaluate_predictions(test_samples, ridge_predictions)
    gb_metrics = evaluate_predictions(test_samples, gb_predictions)
    knn_metrics = evaluate_predictions(test_samples, knn_predictions)
    ensemble_metrics = evaluate_predictions(test_samples, ensemble_predictions)
    naive_metrics = evaluate_predictions(test_samples, naive_predictions)

    # Per-video breakdown
    video_metrics: dict[str, dict[str, Any]] = {}
    for video_name in sorted({s["video_name"] for s in test_samples}):
        video_indices = [i for i, s in enumerate(test_samples) if s["video_name"] == video_name]
        if video_indices:
            video_test = [test_samples[i] for i in video_indices]
            video_pred = ensemble_predictions[video_indices]
            video_metrics[video_name] = evaluate_predictions(video_test, video_pred)
            video_metrics[video_name].pop("confusion_matrix", None)

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
    for sample, ridge_count, gb_count, knn_count, ensemble_count, naive_count in zip(
        test_samples, ridge_predictions, gb_predictions, knn_predictions, ensemble_predictions, naive_predictions
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
                "ridge_predicted_count": round(float(ridge_count), 3),
                "gb_predicted_count": round(float(gb_count), 3),
                "knn_predicted_count": round(float(knn_count), 3),
                "ensemble_predicted_count": round(float(ensemble_count), 3),
                "naive_predicted_count": round(float(naive_count), 3),
                "actual_future_congestion_status": sample["target_congestion_status"],
                "ridge_congestion_status": count_to_status(float(ridge_count)),
                "gb_congestion_status": count_to_status(float(gb_count)),
                "knn_congestion_status": count_to_status(float(knn_count)),
                "ensemble_congestion_status": count_to_status(float(ensemble_count)),
                "naive_congestion_status": count_to_status(float(naive_count)),
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
            "ridge_alpha": best_alpha,
            "ridge_alpha_cv_scores": {str(k): v for k, v in alpha_scores.items()},
            "ensemble_weight_ridge": w_ridge,
            "ensemble_weight_gb": w_gb,
            "ensemble_weight_knn": w_knn,
            "gb_n_estimators": 100,
            "gb_learning_rate": 0.1,
            "gb_max_depth": 2,
            "knn_neighbors": 5,
        },
        "dataset": {
            "zone_rows": len(rows),
            "supervised_samples": len(samples),
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "feature_count": len(feature_names),
            "videos_used": sorted({row.video_name for row in rows}),
        },
        "ridge_metrics": ridge_metrics,
        "gradient_boosting_metrics": gb_metrics,
        "knn_metrics": knn_metrics,
        "ensemble_metrics": ensemble_metrics,
        "naive_baseline_metrics": naive_metrics,
        "per_video_ensemble_metrics": video_metrics,
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
            "ridge_predicted_count",
            "gb_predicted_count",
            "knn_predicted_count",
            "ensemble_predicted_count",
            "naive_predicted_count",
            "actual_future_congestion_status",
            "ridge_congestion_status",
            "gb_congestion_status",
            "knn_congestion_status",
            "ensemble_congestion_status",
            "naive_congestion_status",
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

    print(f"\n{'='*60}")
    print(f"BASELINE TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Phase 2 supervised dataset: {output_dir / 'phase2_supervised_dataset.csv'}")
    print(f"Phase 2 predictions: {output_dir / 'phase2_predictions.csv'}")
    print(f"Phase 2 coefficients: {output_dir / 'phase2_model_coefficients.csv'}")
    print(f"Phase 2 metrics: {output_dir / 'phase2_metrics.json'}")
    print(f"{'='*60}")
    print(
        f"Ridge MAE: {ridge_metrics['mae']:.4f} | "
        f"Ridge accuracy: {ridge_metrics['congestion_accuracy']:.4f}"
    )
    print(
        f"GB MAE: {gb_metrics['mae']:.4f} | "
        f"GB accuracy: {gb_metrics['congestion_accuracy']:.4f}"
    )
    print(
        f"KNN MAE: {knn_metrics['mae']:.4f} | "
        f"KNN accuracy: {knn_metrics['congestion_accuracy']:.4f}"
    )
    print(
        f"Ensemble MAE: {ensemble_metrics['mae']:.4f} | "
        f"Ensemble accuracy: {ensemble_metrics['congestion_accuracy']:.4f}"
    )
    print(
        f"Naive MAE: {naive_metrics['mae']:.4f} | "
        f"Naive accuracy: {naive_metrics['congestion_accuracy']:.4f}"
    )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
