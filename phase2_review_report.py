import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


STATUSES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a review-friendly Phase 2 report from LSTM outputs."
    )
    parser.add_argument(
        "--run-dir",
        default="output/phase2_training/phase2_lstm",
        help="Directory containing phase2_lstm_metrics.json and phase2_lstm_predictions.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional report directory. Defaults to <run-dir>/review_report.",
    )
    return parser.parse_args()


def read_predictions(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "review_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "phase2_lstm_metrics.json"
    predictions_path = run_dir / "phase2_lstm_predictions.csv"
    history_path = run_dir / "phase2_lstm_history.csv"

    with open(metrics_path) as handle:
        metrics = json.load(handle)
    predictions = read_predictions(predictions_path)

    step_errors: defaultdict[int, list[float]] = defaultdict(list)
    zone_errors: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    confusion: Counter[tuple[str, str]] = Counter()
    exact_matches = 0

    for row in predictions:
        step = int(row["forecast_step"])
        actual_count = float(row["actual_future_avg_people_count"])
        predicted_count = float(row["predicted_future_avg_people_count"])
        actual_status = row["actual_future_congestion_status"]
        predicted_status = row["predicted_future_congestion_status"]
        error = abs(predicted_count - actual_count)

        step_errors[step].append(error)
        zone_errors[(row["video_name"], row["zone_number"])].append(error)
        confusion[(actual_status, predicted_status)] += 1
        if actual_status == predicted_status:
            exact_matches += 1

    step_rows = [
        {
            "forecast_step": step,
            "sample_count": len(errors),
            "mean_absolute_error": round(sum(errors) / len(errors), 4),
        }
        for step, errors in sorted(step_errors.items())
    ]
    zone_rows = [
        {
            "video_name": video_name,
            "zone_number": zone_number,
            "sample_count": len(errors),
            "mean_absolute_error": round(sum(errors) / len(errors), 4),
        }
        for (video_name, zone_number), errors in sorted(zone_errors.items())
    ]
    confusion_rows = []
    for actual_status in STATUSES:
        row = {"actual_status": actual_status}
        for predicted_status in STATUSES:
            row[predicted_status] = confusion[(actual_status, predicted_status)]
        confusion_rows.append(row)

    summary_lines = [
        "# Phase 2 Review Summary",
        "",
        "## Model Setup",
        f"- Input root: `{metrics['config']['input_root']}`",
        f"- Lookback window: `{metrics['config']['lookback_seconds']}` seconds",
        f"- Prediction horizon: `{metrics['config']['predict_seconds']}` seconds",
        f"- Hidden size: `{metrics['config']['hidden_size']}`",
        f"- LSTM layers: `{metrics['config']['num_layers']}`",
        f"- Training epochs requested: `{metrics['config']['epochs']}`",
        f"- Device: `{metrics['device']}`",
        "",
        "## Dataset",
        f"- Zone rows loaded: `{metrics['dataset']['zone_rows']}`",
        f"- Total sequences built: `{metrics['dataset']['total_sequences']}`",
        f"- Train sequences: `{metrics['dataset']['train_sequences']}`",
        f"- Validation sequences: `{metrics['dataset']['val_sequences']}`",
        f"- Test sequences: `{metrics['dataset']['test_sequences']}`",
        f"- Videos with sequences: `{', '.join(metrics['dataset'].get('videos_with_sequences', [])) or 'None'}`",
        "",
        "## Test Performance",
        f"- Best validation MAE: `{metrics['best_validation_mae']:.4f}`",
        f"- Test MAE: `{metrics['test_metrics']['mae']:.4f}`",
        f"- Test RMSE: `{metrics['test_metrics']['rmse']:.4f}`",
        f"- Test congestion accuracy: `{metrics['test_metrics']['congestion_accuracy']:.4f}`",
        "",
        "## Derived Review Notes",
        f"- Prediction rows analysed: `{len(predictions)}`",
        f"- Exact congestion-label matches: `{exact_matches}`",
        f"- History file: `{history_path}`",
        f"- Per-step metrics CSV: `{output_dir / 'phase2_step_metrics.csv'}`",
        f"- Confusion matrix CSV: `{output_dir / 'phase2_confusion_matrix.csv'}`",
        f"- Zone summary CSV: `{output_dir / 'phase2_zone_summary.csv'}`",
    ]

    summary_path = output_dir / "phase2_review_summary.md"
    with open(summary_path, "w") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    write_csv(
        output_dir / "phase2_step_metrics.csv",
        ["forecast_step", "sample_count", "mean_absolute_error"],
        step_rows,
    )
    write_csv(
        output_dir / "phase2_zone_summary.csv",
        ["video_name", "zone_number", "sample_count", "mean_absolute_error"],
        zone_rows,
    )
    write_csv(
        output_dir / "phase2_confusion_matrix.csv",
        ["actual_status"] + STATUSES,
        confusion_rows,
    )

    print(f"Review summary: {summary_path}")
    print(f"Step metrics: {output_dir / 'phase2_step_metrics.csv'}")
    print(f"Zone summary: {output_dir / 'phase2_zone_summary.csv'}")
    print(f"Confusion matrix: {output_dir / 'phase2_confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
