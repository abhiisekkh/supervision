import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from phase2_lstm_train import (
    EncoderDecoderLSTM,
    FEATURE_NAMES,
    SequenceDataset,
    build_sequences,
    evaluate_counts,
    invert_scale,
    load_zone_rows,
    metadata_to_rows,
    resolve_device,
    sanitize_count_predictions,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 LSTM inference using a saved checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default="output/phase2_training/phase2_lstm/phase2_lstm_checkpoint.pt",
        help="Path to the saved LSTM checkpoint.",
    )
    parser.add_argument(
        "--input-root",
        default="output/phase1_processed",
        help="Phase 1 root directory, or one specific processed video folder.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/phase2_training/phase2_lstm_inference",
        help="Directory where inference CSV/JSON files are written.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or mps.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint at {path} is missing model weights.")
    return checkpoint


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)

    config = checkpoint.get("config", {})
    lookback_seconds = int(config.get("lookback_seconds", 8))
    predict_seconds = int(config.get("predict_seconds", 3))
    hidden_size = int(config.get("hidden_size", 64))
    num_layers = int(config.get("num_layers", 2))
    dropout = float(config.get("dropout", 0.2))

    rows = load_zone_rows(Path(args.input_root))
    samples = build_sequences(
        rows=rows,
        lookback_seconds=lookback_seconds,
        predict_seconds=predict_seconds,
    )
    if not samples:
        raise ValueError(
            "No valid sequences found for inference. Check the processed video folder "
            "or use a shorter lookback/predict window during training."
        )

    dataset = SequenceDataset(
        samples=samples,
        feature_mean=torch.tensor(checkpoint["feature_mean"]).numpy(),
        feature_std=torch.tensor(checkpoint["feature_std"]).numpy(),
        target_mean=float(checkpoint["target_mean"]),
        target_std=float(checkpoint["target_std"]),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = EncoderDecoderLSTM(
        input_size=len(FEATURE_NAMES),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        prediction_horizon=predict_seconds,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loss_fn = nn.MSELoss()
    losses: list[float] = []
    predictions_list = []
    targets_list = []
    metas: list[dict[str, Any]] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        with torch.no_grad():
            prediction = model(x, target=None, teacher_forcing_ratio=0.0)
            loss = loss_fn(prediction, y)
        losses.append(float(loss.item()))
        predictions_list.append(prediction.detach().cpu())
        targets_list.append(y.detach().cpu())

        batch_meta = batch["meta"]
        keys = list(batch_meta.keys())
        batch_size = len(batch_meta[keys[0]]) if keys else 0
        metas.extend(
            [{key: batch_meta[key][index] for key in keys} for index in range(batch_size)]
        )

    predictions = torch.cat(predictions_list, dim=0).numpy()
    targets = torch.cat(targets_list, dim=0).numpy()
    predictions_unscaled = sanitize_count_predictions(
        invert_scale(predictions, float(checkpoint["target_mean"]), float(checkpoint["target_std"]))
    )
    targets_unscaled = invert_scale(
        targets, float(checkpoint["target_mean"]), float(checkpoint["target_std"])
    )
    metrics = evaluate_counts(predictions_unscaled, targets_unscaled)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "phase2_lstm_inference_predictions.csv"
    metrics_path = output_dir / "phase2_lstm_inference_metrics.json"

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
        metadata_to_rows(metas, predictions_unscaled, targets_unscaled),
    )

    payload = {
        "checkpoint": str(checkpoint_path),
        "input_root": str(Path(args.input_root)),
        "device": str(device),
        "sequence_count": len(samples),
        "videos_with_sequences": sorted({sample.video_name for sample in samples}),
        "average_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "metrics": metrics,
    }
    with open(metrics_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Inference predictions: {predictions_path}")
    print(f"Inference metrics: {metrics_path}")
    print(
        f"Inference MAE: {metrics['mae']:.3f} | "
        f"Inference congestion accuracy: {metrics['congestion_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
