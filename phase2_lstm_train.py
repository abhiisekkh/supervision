import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


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


@dataclass
class SequenceSample:
    video_name: str
    zone_number: int
    input_start_sec: int
    input_end_sec: int
    target_start_sec: int
    target_end_sec: int
    input_features: np.ndarray
    target_counts: np.ndarray


class SequenceDataset(Dataset):
    def __init__(
        self,
        samples: list[SequenceSample],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        target_mean: float,
        target_std: float,
    ) -> None:
        self.samples = samples
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        x = (sample.input_features - self.feature_mean) / self.feature_std
        y = (sample.target_counts - self.target_mean) / self.target_std
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "meta": {
                "video_name": sample.video_name,
                "zone_number": sample.zone_number,
                "input_start_sec": sample.input_start_sec,
                "input_end_sec": sample.input_end_sec,
                "target_start_sec": sample.target_start_sec,
                "target_end_sec": sample.target_end_sec,
            },
        }


class EncoderDecoderLSTM(nn.Module):
    """Original encoder-decoder LSTM (kept for backward checkpoint compatibility)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        prediction_horizon: int,
    ) -> None:
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        _, (hidden, cell) = self.encoder(x)

        decoder_input = x[:, -1:, 0:1]
        outputs: list[torch.Tensor] = []

        for step in range(self.prediction_horizon):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            step_prediction = self.output(decoder_output)
            outputs.append(step_prediction)

            if target is not None and random.random() < teacher_forcing_ratio:
                decoder_input = target[:, step : step + 1].unsqueeze(-1)
            else:
                decoder_input = step_prediction

        return torch.cat(outputs, dim=1).squeeze(-1)


class AttentionEncoderDecoderLSTM(nn.Module):
    """Enhanced LSTM with bidirectional encoder, multi-head attention, layer
    normalization, and residual connections for maximum accuracy."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        prediction_horizon: int,
        num_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection with layer norm
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Bidirectional encoder
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.encoder_norm = nn.LayerNorm(hidden_size * 2)

        # Project bidirectional encoder output to decoder hidden size
        self.encoder_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_projection = nn.Linear(hidden_size * 2, hidden_size)

        # Decoder
        self.decoder = nn.LSTM(
            input_size=1 + hidden_size,  # prediction + context from attention
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder_norm = nn.LayerNorm(hidden_size)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout * 0.5,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_size)

        # Output layers with residual
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        # Project input features
        x_projected = self.input_projection(x)

        # Bidirectional encoder
        encoder_outputs, (hidden, cell) = self.encoder(x_projected)
        encoder_outputs = self.encoder_norm(encoder_outputs)

        # Project encoder outputs for attention keys/values
        encoder_projected = self.encoder_projection(encoder_outputs)

        # Merge bidirectional hidden states for decoder init
        # hidden shape: (num_layers*2, batch, hidden) -> (num_layers, batch, hidden)
        hidden_fwd = hidden[0::2]  # forward layers
        hidden_bwd = hidden[1::2]  # backward layers
        hidden_merged = self.hidden_projection(
            torch.cat([hidden_fwd, hidden_bwd], dim=-1)
        )

        cell_fwd = cell[0::2]
        cell_bwd = cell[1::2]
        cell_merged = self.cell_projection(
            torch.cat([cell_fwd, cell_bwd], dim=-1)
        )

        # Decoder auto-regressive loop
        decoder_hidden = hidden_merged
        decoder_cell = cell_merged
        decoder_input_val = x[:, -1:, 0:1]  # last avg_people_count
        outputs: list[torch.Tensor] = []

        for step in range(self.prediction_horizon):
            # Attention: query = previous decoder hidden (top layer)
            query = decoder_hidden[-1:].permute(1, 0, 2)  # (batch, 1, hidden)
            context, _ = self.attention(
                query, encoder_projected, encoder_projected
            )
            context = self.attention_norm(context + query)  # residual

            # Concatenate prediction input with attention context
            decoder_in = torch.cat(
                [decoder_input_val, context], dim=-1
            )  # (batch, 1, 1+hidden)

            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                decoder_in, (decoder_hidden, decoder_cell)
            )
            decoder_output = self.decoder_norm(decoder_output)

            # Output with residual from context
            combined = torch.cat([decoder_output, context], dim=-1)
            step_prediction = self.output_mlp(combined)  # (batch, 1, 1)
            outputs.append(step_prediction)

            if target is not None and random.random() < teacher_forcing_ratio:
                decoder_input_val = target[:, step : step + 1].unsqueeze(-1)
            else:
                decoder_input_val = step_prediction

        return torch.cat(outputs, dim=1).squeeze(-1)


class EMAModel:
    """Exponential Moving Average of model parameters for better generalization."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 LSTM Seq2Seq training for crowd-flow forecasting."
    )
    parser.add_argument(
        "--input-root",
        default="output/phase1_processed",
        help="Phase 1 root directory, or one specific processed video folder.",
    )
    parser.add_argument(
        "--lookback-seconds",
        type=int,
        default=30,
        help="Number of previous zone windows used as input sequence length (more=better context).",
    )
    parser.add_argument(
        "--predict-seconds",
        type=int,
        default=10,
        help="Number of future zone windows predicted by the model (prediction horizon).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Per-series train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Per-series validation split ratio.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs (more epochs with early stopping for best convergence).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="AdamW learning rate (lower for attention model stability).",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="LSTM hidden size (larger capacity for attention model).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of LSTM layers (deeper network with attention).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout used inside stacked LSTMs (higher for regularization).",
    )
    parser.add_argument(
        "--teacher-forcing",
        type=float,
        default=0.5,
        help="Initial teacher forcing ratio during training.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience on validation MAE (longer for thorough exploration).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/phase2_training/phase2_lstm",
        help="Directory for checkpoints, metrics, and predictions.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay for regularization.",
    )
    parser.add_argument(
        "--use-attention",
        action="store_true",
        default=True,
        help="Use the enhanced attention-based LSTM model (default: True).",
    )
    parser.add_argument(
        "--no-attention",
        action="store_true",
        default=False,
        help="Use the original encoder-decoder LSTM instead of attention model.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="Exponential moving average decay for model weights.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_to_status(count: float) -> str:
    count = max(0.0, float(count))
    if count < 5:
        return "LOW"
    if count < 15:
        return "MEDIUM"
    if count < 30:
        return "HIGH"
    return "CRITICAL"


def encode_zone_row(row: dict[str, str]) -> dict[str, float]:
    direction_x, direction_y = DIRECTION_TO_VECTOR.get(
        row["dominant_direction"], (0.0, 0.0)
    )
    return {
        "avg_people_count": float(row["avg_people_count"]),
        "max_people_count": float(row["max_people_count"]),
        "unique_track_ids": float(row["unique_track_ids"]),
        "avg_speed_px_per_sec": float(row["avg_speed_px_per_sec"]),
        "mean_flow_dx_per_frame": float(row["mean_flow_dx_per_frame"]),
        "mean_flow_dy_per_frame": float(row["mean_flow_dy_per_frame"]),
        "mean_optical_flow_dx_per_frame": float(
            row["mean_optical_flow_dx_per_frame"]
        ),
        "mean_optical_flow_dy_per_frame": float(
            row["mean_optical_flow_dy_per_frame"]
        ),
        "inflow_count": float(row["inflow_count"]),
        "outflow_count": float(row["outflow_count"]),
        "direction_x": direction_x,
        "direction_y": direction_y,
        "congestion_level": float(STATUS_TO_INT[row["congestion_status"]]),
    }


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
                        values=encode_zone_row(row),
                    )
                )
    return rows


def build_sequences(
    rows: list[ZoneRow], lookback_seconds: int, predict_seconds: int
) -> list[SequenceSample]:
    grouped: dict[tuple[str, int], list[ZoneRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.video_name, row.zone_number)].append(row)

    samples: list[SequenceSample] = []
    for (video_name, zone_number), sequence in grouped.items():
        ordered = sorted(sequence, key=lambda row: row.window_start_sec)
        if len(ordered) < lookback_seconds + predict_seconds:
            continue

        for start_index in range(0, len(ordered) - lookback_seconds - predict_seconds + 1):
            history = ordered[start_index : start_index + lookback_seconds]
            future = ordered[
                start_index + lookback_seconds : start_index + lookback_seconds + predict_seconds
            ]

            expected_seconds = list(
                range(history[0].window_start_sec, future[-1].window_start_sec + 1)
            )
            actual_seconds = [row.window_start_sec for row in history + future]
            if actual_seconds != expected_seconds:
                continue

            input_features = np.array(
                [[row.values[name] for name in FEATURE_NAMES] for row in history],
                dtype=np.float32,
            )
            target_counts = np.array(
                [row.values["avg_people_count"] for row in future], dtype=np.float32
            )

            samples.append(
                SequenceSample(
                    video_name=video_name,
                    zone_number=zone_number,
                    input_start_sec=history[0].window_start_sec,
                    input_end_sec=history[-1].window_start_sec,
                    target_start_sec=future[0].window_start_sec,
                    target_end_sec=future[-1].window_start_sec,
                    input_features=input_features,
                    target_counts=target_counts,
                )
            )
    return samples


def split_single_series(
    ordered: list[SequenceSample], train_ratio: float, val_ratio: float
) -> tuple[list[SequenceSample], list[SequenceSample], list[SequenceSample]]:
    length = len(ordered)
    if length == 0:
        return [], [], []
    if length == 1:
        return ordered[:], [], []
    if length == 2:
        return ordered[:1], [], ordered[1:]
    if length == 3:
        return ordered[:1], ordered[1:2], ordered[2:]

    train_count = max(1, int(length * train_ratio))
    val_count = max(1, int(length * val_ratio))

    if train_count + val_count >= length:
        overflow = (train_count + val_count) - (length - 1)
        while overflow > 0 and train_count > 1:
            train_count -= 1
            overflow -= 1
        while overflow > 0 and val_count > 1:
            val_count -= 1
            overflow -= 1

    test_count = length - train_count - val_count
    if test_count <= 0:
        if train_count > val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            train_count = max(1, train_count - 1)
        test_count = length - train_count - val_count

    train_samples = ordered[:train_count]
    val_samples = ordered[train_count : train_count + val_count]
    test_samples = ordered[train_count + val_count : train_count + val_count + test_count]
    return train_samples, val_samples, test_samples


def rebalance_global_splits(
    train_samples: list[SequenceSample],
    val_samples: list[SequenceSample],
    test_samples: list[SequenceSample],
) -> tuple[list[SequenceSample], list[SequenceSample], list[SequenceSample]]:
    if not val_samples and len(train_samples) >= 2:
        val_samples.append(train_samples.pop())
    if not test_samples:
        if len(train_samples) >= 2:
            test_samples.append(train_samples.pop())
        elif val_samples:
            test_samples.append(val_samples[-1])
    return train_samples, val_samples, test_samples


def split_sequences(
    samples: list[SequenceSample], train_ratio: float, val_ratio: float
) -> tuple[list[SequenceSample], list[SequenceSample], list[SequenceSample]]:
    # Group by video first, then by zone
    video_grouped: dict[str, list[SequenceSample]] = defaultdict(list)
    for sample in samples:
        video_grouped[sample.video_name].append(sample)
    
    train_samples: list[SequenceSample] = []
    val_samples: list[SequenceSample] = []
    test_samples: list[SequenceSample] = []

    # Ensure each video contributes to each split (stratified by video)
    for video_name in sorted(video_grouped):
        video_samples = video_grouped[video_name]
        
        # Further group by zone within this video
        zone_grouped: dict[int, list[SequenceSample]] = defaultdict(list)
        for sample in video_samples:
            zone_grouped[sample.zone_number].append(sample)
        
        # Split each zone
        for zone_number in sorted(zone_grouped):
            ordered = sorted(zone_grouped[zone_number], key=lambda sample: sample.input_start_sec)
            series_train, series_val, series_test = split_single_series(
                ordered, train_ratio=train_ratio, val_ratio=val_ratio
            )
            train_samples.extend(series_train)
            val_samples.extend(series_val)
            test_samples.extend(series_test)

    train_samples, val_samples, test_samples = rebalance_global_splits(train_samples, val_samples, test_samples)
    
    # Ensure all videos are represented in test set (stratification guarantee)
    test_videos = set(s.video_name for s in test_samples)
    all_videos = set(s.video_name for s in samples)
    missing_videos = all_videos - test_videos
    
    for missing_video in sorted(missing_videos):
        # Find a sample from this video in train or val and move to test
        moved = False
        for source_list in [val_samples, train_samples]:
            for i, sample in enumerate(source_list):
                if sample.video_name == missing_video:
                    test_samples.append(source_list.pop(i))
                    moved = True
                    break
            if moved:
                break
    
    return train_samples, val_samples, test_samples


def compute_normalization_stats(
    samples: list[SequenceSample],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.concatenate([sample.input_features for sample in samples], axis=0)
    y = np.concatenate([sample.target_counts for sample in samples], axis=0)

    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    target_mean = float(y.mean())
    target_std = float(y.std())
    if target_std < 1e-6:
        target_std = 1.0

    return feature_mean, feature_std, target_mean, target_std


def run_epoch(
    model: EncoderDecoderLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
    device: torch.device,
    teacher_forcing_ratio: float,
) -> tuple[float, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    training = optimizer is not None
    model.train(training)

    losses: list[float] = []
    predictions_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.set_grad_enabled(training):
            prediction = model(
                x,
                target=y if training else None,
                teacher_forcing_ratio=teacher_forcing_ratio if training else 0.0,
            )
            loss = loss_fn(prediction, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        losses.append(float(loss.item()))
        predictions_list.append(prediction.detach().cpu().numpy())
        targets_list.append(y.detach().cpu().numpy())
        batch_meta = batch["meta"]
        if isinstance(batch_meta, dict):
            keys = list(batch_meta.keys())
            batch_size = len(batch_meta[keys[0]]) if keys else 0
            metas.extend(
                [
                    {key: batch_meta[key][index] for key in keys}
                    for index in range(batch_size)
                ]
            )
        else:
            metas.extend(batch_meta)

    predictions = np.concatenate(predictions_list, axis=0) if predictions_list else np.empty((0, 0))
    targets = np.concatenate(targets_list, axis=0) if targets_list else np.empty((0, 0))
    average_loss = float(np.mean(losses)) if losses else 0.0
    return average_loss, predictions, targets, metas


def invert_scale(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


def sanitize_count_predictions(values: np.ndarray) -> np.ndarray:
    return np.clip(values, a_min=0.0, a_max=None)


def evaluate_counts(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    if predictions.size == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "step_mae": [],
            "congestion_accuracy": 0.0,
        }

    safe_predictions = sanitize_count_predictions(predictions)
    mae = float(np.mean(np.abs(safe_predictions - targets)))
    rmse = float(np.sqrt(np.mean((safe_predictions - targets) ** 2)))
    step_mae = np.mean(np.abs(safe_predictions - targets), axis=0).tolist()

    actual_status = [[count_to_status(value) for value in row] for row in targets]
    predicted_status = [
        [count_to_status(value) for value in row] for row in safe_predictions
    ]
    total = sum(len(row) for row in actual_status)
    matches = sum(
        1
        for actual_row, predicted_row in zip(actual_status, predicted_status)
        for actual_value, predicted_value in zip(actual_row, predicted_row)
        if actual_value == predicted_value
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "step_mae": [float(value) for value in step_mae],
        "congestion_accuracy": float(matches / total) if total else 0.0,
    }


def metadata_to_rows(
    metas: list[dict[str, Any]],
    predictions: np.ndarray,
    targets: np.ndarray,
) -> list[dict[str, Any]]:
    def normalize_meta_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    rows: list[dict[str, Any]] = []
    for meta, predicted_seq, target_seq in zip(
        metas, sanitize_count_predictions(predictions), targets
    ):
        normalized_meta = {
            key: normalize_meta_value(value) for key, value in meta.items()
        }
        for step_index, (predicted_value, target_value) in enumerate(
            zip(predicted_seq, target_seq),
            start=1,
        ):
            target_second = int(normalized_meta["target_start_sec"]) + (step_index - 1)
            rows.append(
                {
                    "video_name": normalized_meta["video_name"],
                    "zone_number": int(normalized_meta["zone_number"]),
                    "input_start_sec": int(normalized_meta["input_start_sec"]),
                    "input_end_sec": int(normalized_meta["input_end_sec"]),
                    "target_second": target_second,
                    "forecast_step": step_index,
                    "actual_future_avg_people_count": round(float(target_value), 4),
                    "predicted_future_avg_people_count": round(float(predicted_value), 4),
                    "actual_future_congestion_status": count_to_status(float(target_value)),
                    "predicted_future_congestion_status": count_to_status(
                        float(predicted_value)
                    ),
                }
            )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    rows = load_zone_rows(Path(args.input_root))
    samples = build_sequences(
        rows=rows,
        lookback_seconds=max(1, args.lookback_seconds),
        predict_seconds=max(1, args.predict_seconds),
    )
    if not samples:
        raise ValueError(
            "No valid sequences found. You may need longer videos or a shorter lookback/predict window."
        )

    train_samples, val_samples, test_samples = split_sequences(
        samples=samples,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
    )
    if not train_samples:
        raise ValueError("No training samples available for LSTM training.")
    if not val_samples or not test_samples:
        raise ValueError(
            "Not enough non-overlapping validation/test sequences were created. "
            "Use more videos or shorten the lookback/predict windows."
        )

    feature_mean, feature_std, target_mean, target_std = compute_normalization_stats(
        train_samples
    )

    train_dataset = SequenceDataset(
        samples=train_samples,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    val_dataset = SequenceDataset(
        samples=val_samples,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    test_dataset = SequenceDataset(
        samples=test_samples,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    use_attention = args.use_attention and not args.no_attention
    if use_attention:
        print("Using AttentionEncoderDecoderLSTM (bidirectional + multi-head attention)")
        model = AttentionEncoderDecoderLSTM(
            input_size=len(FEATURE_NAMES),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            prediction_horizon=args.predict_seconds,
            num_attention_heads=4,
        ).to(device)
    else:
        print("Using original EncoderDecoderLSTM")
        model = EncoderDecoderLSTM(
            input_size=len(FEATURE_NAMES),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            prediction_horizon=args.predict_seconds,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    loss_fn = nn.HuberLoss(delta=1.0)
    ema = EMAModel(model, decay=args.ema_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "phase2_lstm_checkpoint.pt"

    best_val_mae = math.inf
    best_state: dict[str, Any] | None = None
    patience_counter = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        teacher_forcing_ratio = max(
            0.1, float(args.teacher_forcing) * (1.0 - ((epoch - 1) / args.epochs))
        )
        train_loss, train_pred, train_target, _train_meta = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        ema.update(model)
        scheduler.step()

        # Evaluate with EMA weights for more stable validation
        ema.apply_shadow(model)
        val_loss, val_pred, val_target, _val_meta = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            loss_fn=loss_fn,
            device=device,
            teacher_forcing_ratio=0.0,
        )
        ema.restore(model)

        train_pred_unscaled = sanitize_count_predictions(
            invert_scale(train_pred, target_mean, target_std)
        )
        train_target_unscaled = invert_scale(train_target, target_mean, target_std)
        val_pred_unscaled = sanitize_count_predictions(
            invert_scale(val_pred, target_mean, target_std)
        )
        val_target_unscaled = invert_scale(val_target, target_mean, target_std)

        train_metrics = evaluate_counts(train_pred_unscaled, train_target_unscaled)
        val_metrics = evaluate_counts(val_pred_unscaled, val_target_unscaled)

        current_lr = optimizer.param_groups[0]["lr"]
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "train_mae": round(train_metrics["mae"], 6),
                "val_mae": round(val_metrics["mae"], 6),
                "train_rmse": round(train_metrics["rmse"], 6),
                "val_rmse": round(val_metrics["rmse"], 6),
                "teacher_forcing_ratio": round(teacher_forcing_ratio, 4),
                "learning_rate": round(current_lr, 8),
            }
        )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{args.epochs} | "
                f"train_mae={train_metrics['mae']:.4f} val_mae={val_metrics['mae']:.4f} | "
                f"train_acc={train_metrics['congestion_accuracy']:.3f} "
                f"val_acc={val_metrics['congestion_accuracy']:.3f} | "
                f"lr={current_lr:.2e}"
            )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            patience_counter = 0
            # Save EMA weights as best (they generalize better)
            ema.apply_shadow(model)
            best_state = {
                "model_state_dict": model.state_dict(),
                "feature_mean": feature_mean.tolist(),
                "feature_std": feature_std.tolist(),
                "target_mean": target_mean,
                "target_std": target_std,
                "config": vars(args),
                "model_type": "attention" if use_attention else "original",
            }
            torch.save(best_state, checkpoint_path)
            ema.restore(model)
            print(f"  *** New best val MAE: {best_val_mae:.4f} (saved) ***")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    if best_state is None:
        raise RuntimeError("Training finished without producing a valid checkpoint.")

    model.load_state_dict(best_state["model_state_dict"])
    test_loss, test_pred, test_target, test_meta = run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        loss_fn=loss_fn,
        device=device,
        teacher_forcing_ratio=0.0,
    )
    test_pred_unscaled = sanitize_count_predictions(
        invert_scale(test_pred, target_mean, target_std)
    )
    test_target_unscaled = invert_scale(test_target, target_mean, target_std)
    test_metrics = evaluate_counts(test_pred_unscaled, test_target_unscaled)

    history_path = output_dir / "phase2_lstm_history.csv"
    predictions_path = output_dir / "phase2_lstm_predictions.csv"
    metrics_path = output_dir / "phase2_lstm_metrics.json"

    write_csv(
        history_path,
        [
            "epoch",
            "train_loss",
            "val_loss",
            "train_mae",
            "val_mae",
            "train_rmse",
            "val_rmse",
            "teacher_forcing_ratio",
            "learning_rate",
        ],
        history_rows,
    )
    write_csv(
        predictions_path,
        [
            "video_name",
            "zone_number",
            "input_start_sec",
            "input_end_sec",
            "target_second",
            "forecast_step",
            "actual_future_avg_people_count",
            "predicted_future_avg_people_count",
            "actual_future_congestion_status",
            "predicted_future_congestion_status",
        ],
        metadata_to_rows(test_meta, test_pred_unscaled, test_target_unscaled),
    )

    metrics = {
        "config": vars(args),
        "device": str(device),
        "model_type": "attention" if use_attention else "original",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "dataset": {
            "zone_rows": len(rows),
            "total_sequences": len(samples),
            "train_sequences": len(train_samples),
            "val_sequences": len(val_samples),
            "test_sequences": len(test_samples),
            "videos_with_zone_rows": sorted({row.video_name for row in rows}),
            "videos_with_sequences": sorted({sample.video_name for sample in samples}),
            "series_with_sequences": len(
                {(sample.video_name, sample.zone_number) for sample in samples}
            ),
            "feature_names": FEATURE_NAMES,
        },
        "best_validation_mae": best_val_mae,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "normalization": {
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "target_mean": target_mean,
            "target_std": target_std,
        },
    }
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"\n{'='*60}")
    print(f"LSTM TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Model type: {'Attention LSTM' if use_attention else 'Original LSTM'}")
    print(f"Parameters: {trainable_params:,}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(f"{'='*60}")
    print(
        f"Best validation MAE: {best_val_mae:.4f} | "
        f"Test MAE: {test_metrics['mae']:.4f} | "
        f"Test RMSE: {test_metrics['rmse']:.4f} | "
        f"Test congestion accuracy: {test_metrics['congestion_accuracy']:.4f}"
    )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
