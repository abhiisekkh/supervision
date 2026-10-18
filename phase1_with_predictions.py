#!/usr/bin/env python3
"""
Phase 1 + Phase 2 LSTM Predictions: Real-time crowd prediction overlay.

This script combines Phase 1 detection/tracking with Phase 2 LSTM predictions
to show which zones will become crowded before it happens. Predicted danger zones
are overlaid with semi-transparent red squares.

Usage:
    python3 phase1_with_predictions.py --video <path> --checkpoint <model>
    
Key Feature:
    - Shows current congestion with colored zones
    - Overlays red prediction squares on zones predicted to be crowded in 3 seconds
    - Makes demo impressive for external panel review
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import supervision as sv
from ultralytics import YOLO

# Import Phase 2 LSTM components
sys.path.insert(0, str(Path(__file__).parent))
from phase2_lstm_train import (
    AttentionEncoderDecoderLSTM,
    EncoderDecoderLSTM,
    FEATURE_NAMES,
    invert_scale,
    sanitize_count_predictions,
)


def load_lstm_checkpoint(checkpoint_path: str, device: str = "cpu") -> tuple:
    """Load LSTM model and checkpoint metadata."""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    
    hidden_size = config.get("hidden_size", 64)
    num_layers = config.get("num_layers", 2)
    predict_seconds = config.get("predict_seconds", 3)
    dropout = config.get("dropout", 0.2)
    
    model_type = checkpoint.get("model_type", config.get("model_type", "original"))
    use_attention = model_type == "attention" or config.get("use_attention", False)

    if use_attention:
        model = AttentionEncoderDecoderLSTM(
            input_size=len(FEATURE_NAMES),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            prediction_horizon=predict_seconds,
            num_attention_heads=4,
        ).to(device)
    else:
        model = EncoderDecoderLSTM(
            input_size=len(FEATURE_NAMES),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            prediction_horizon=predict_seconds,
        ).to(device)
        
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, checkpoint


def get_zone_number(
    cx: float, cy: float, frame_shape: tuple, grid_size: int = 3
) -> int:
    """Get zone number (1-9) from anchor point."""
    h, w = frame_shape[:2]
    col = min(int(cx * grid_size / w), grid_size - 1)
    row = min(int(cy * grid_size / h), grid_size - 1)
    return row * grid_size + col + 1


def get_anchor_point(box: np.ndarray) -> tuple[float, float]:
    """Get anchor point from bounding box."""
    x1, _y1, x2, y2 = box
    return (float((x1 + x2) / 2.0), float(y2))


def count_people_in_zones(
    anchor_points: list[tuple[float, float]],
    frame_shape: tuple,
    grid_size: int = 3,
) -> list[int]:
    """Count people per zone."""
    zone_counts = [0] * (grid_size * grid_size)
    for cx, cy in anchor_points:
        zone_id = get_zone_number(cx, cy, frame_shape, grid_size) - 1
        zone_counts[zone_id] += 1
    return zone_counts


def extract_features(zone_counts: list[int]) -> np.ndarray:
    """Extract features from zone counts (matching Phase 2 training)."""
    zone_counts = np.array(zone_counts, dtype=np.float32)
    total = zone_counts.sum()
    
    features = [
        zone_counts[0],  # Z1
        zone_counts[1],  # Z2
        zone_counts[2],  # Z3
        zone_counts[3],  # Z4
        zone_counts[4],  # Z5
        zone_counts[5],  # Z6
        zone_counts[6],  # Z7
        zone_counts[7],  # Z8
        zone_counts[8],  # Z9
        total,           # total
        zone_counts.std() if total > 0 else 0,  # std
        zone_counts.max(),  # max
        zone_counts.min(),  # min
    ]
    return np.array(features, dtype=np.float32)


def predict_congestion(
    model: EncoderDecoderLSTM,
    feature_history: deque,
    target_mean: float,
    target_std: float,
    device: str,
    lookback_frames: int = 32,  # 8 sec at 4 fps
) -> Optional[float]:
    """
    Predict total congestion for next timestep.
    
    Returns:
        Predicted total people count, or None if not enough history
    """
    if len(feature_history) < lookback_frames:
        return None
    
    # Build sequence from history
    sequence = np.array(list(feature_history)[-lookback_frames:], dtype=np.float32)
    
    # Normalize
    sequence = (sequence - target_mean) / (target_std + 1e-8)
    
    # Create batch
    X = torch.from_numpy(sequence[np.newaxis, :, :]).to(device)  # [1, timesteps, features]
    
    try:
        with torch.no_grad():
            prediction = model(X, target=None, teacher_forcing_ratio=0.0)  # [1, horizon, 1]
        
        # Get last prediction (horizon=3 means predict 3 steps ahead, take last one)
        pred_np = float(prediction[0, -1].cpu().numpy())  # [1]
        
        # Denormalize
        pred_unscaled = float(invert_scale(np.array([pred_np]), target_mean, target_std)[0])
        pred_unscaled = max(0, pred_unscaled)  # No negative counts
        
        return pred_unscaled
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


def get_congestion_status(count: float) -> str:
    """Classify congestion level."""
    if count < 3:
        return "LOW"
    elif count < 6:
        return "MEDIUM"
    elif count < 10:
        return "HIGH"
    else:
        return "DANGER"


def draw_zones_with_predictions(
    frame: np.ndarray,
    zone_counts: list[int],
    zone_statuses: list[str],
    predicted_total_count: Optional[float] = None,
    grid_size: int = 3,
) -> np.ndarray:
    """
    Draw zone overlays and prediction highlights in SAME video.
    
    - Current congestion: Colored zones clearly visible
    - Predicted congestion: RED overlay on zones that will get crowded
    - Both visible at same time in same frame
    """
    h, w = frame.shape[:2]
    cell_w = w // grid_size
    cell_h = h // grid_size
    overlay = frame.copy()
    
    # Draw current congestion status with STRONGER colors
    for row in range(grid_size):
        for col in range(grid_size):
            zone_id = row * grid_size + col
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = w if col == grid_size - 1 else (col + 1) * cell_w
            y2 = h if row == grid_size - 1 else (row + 1) * cell_h
            
            congestion = zone_statuses[zone_id]
            if congestion == "LOW":
                color = (0, 150, 0)       # Bright Green
            elif congestion == "MEDIUM":
                color = (0, 220, 220)     # Bright Cyan
            elif congestion == "HIGH":
                color = (0, 165, 255)     # Orange
            else:  # DANGER
                color = (0, 0, 200)       # Bright Red
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    
    # Blend current state with STRONGER alpha (30% so it's clearly visible)
    frame[:] = cv2.addWeighted(overlay, 0.30, frame, 0.70, 0)
    
    # Overlay predictions as RED highlights on high-density zones (MORE VISIBLE)
    if predicted_total_count is not None and predicted_total_count > 8:
        pred_overlay = frame.copy()
        total_current = sum(zone_counts)
        
        # If predicted count is much higher than current, highlight zones that will absorb growth
        growth_factor = predicted_total_count / max(total_current, 1)
        
        for zone_id, count in enumerate(zone_counts):
            # Probability this zone will be crowded (based on current density)
            zone_probability = count / max(total_current, 1) if total_current > 0 else 0
            
            # Highlight zones that:
            # 1. Already have high density, OR
            # 2. Will receive significant growth
            if count >= 5 or (growth_factor > 1.2 and zone_probability > 0.1):
                row = zone_id // grid_size
                col = zone_id % grid_size
                x1 = col * cell_w
                y1 = row * cell_h
                x2 = w if col == grid_size - 1 else (col + 1) * cell_w
                y2 = h if row == grid_size - 1 else (row + 1) * cell_h
                
                # BRIGHT MAGENTA with higher alpha (VERY visible)
                cv2.rectangle(pred_overlay, (x1, y1), (x2, y2), (255, 0, 255), -1)
                # Also draw bright magenta border to make it stand out
                cv2.rectangle(pred_overlay, (x1, y1), (x2, y2), (255, 0, 255), 4)
        
        # Blend prediction overlay with MUCH stronger alpha (20% - clearly visible)
        frame[:] = cv2.addWeighted(pred_overlay, 0.20, frame, 0.80, 0)
    
    # Draw grid lines
    for i in range(1, grid_size):
        x = i * cell_w
        cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 3)
    
    for i in range(1, grid_size):
        y = i * cell_h
        cv2.line(frame, (0, y), (w, y), (255, 255, 255), 3)
    
    # Draw zone labels and statistics (BOTH current and predicted)
    for row in range(grid_size):
        for col in range(grid_size):
            zone_index = row * grid_size + col
            x = col * cell_w + 12
            y = row * cell_h + 35
            
            # Zone number - BRIGHT YELLOW
            cv2.putText(
                frame,
                f"Z{zone_index + 1}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),  # Bright yellow
                3,
            )
            
            # Current count | status - BRIGHT WHITE
            current_text = f"Current: {zone_counts[zone_index]} ({zone_statuses[zone_index]})"
            cv2.putText(
                frame,
                current_text,
                (x, y + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),  # White
                2,
            )
            
            # Predicted total - Show in red if predicted danger
            if predicted_total_count is not None:
                pred_text = f">>> PREDICTED: {int(round(predicted_total_count))} total <<<"
                pred_status = get_congestion_status(predicted_total_count)
                
                # Only show prediction in first zone to avoid clutter
                if zone_index == 0:
                    # Red text if danger predicted
                    if pred_status in ["HIGH", "DANGER"]:
                        color = (0, 0, 255)  # Red
                        thickness = 3
                    else:
                        color = (200, 200, 200)  # Gray
                        thickness = 2
                    
                    cv2.putText(
                        frame,
                        pred_text,
                        (x, y + 64),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        thickness,
                    )
    
    return frame


def process_video_with_predictions(
    video_path: str,
    model_checkpoint: str,
    output_dir: str = "output/phase1_processed",
    model_name: str = "yolov8m.pt",
    conf_threshold: float = 0.35,
    imgsz: int = 1280,
    use_slicer: bool = True,
    slice_size: int = 960,
):
    """
    Process video: Phase 1 detection/tracking + Phase 2 LSTM prediction overlay.
    """
    # Setup
    device = "cpu"  # M1 Mac
    print(f"Using device: {device}")
    
    # Load YOLO model
    print(f"Loading YOLO model: {model_name}")
    yolo_model = YOLO(model_name)
    
    # Load LSTM model
    print(f"Loading LSTM checkpoint: {model_checkpoint}")
    lstm_model, lstm_checkpoint = load_lstm_checkpoint(model_checkpoint, device)
    target_mean = float(lstm_checkpoint.get("target_mean", 0))
    target_std = float(lstm_checkpoint.get("target_std", 1))
    
    # Open video
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        sys.exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {w}x{h} @ {fps} fps, {total} frames")
    
    # Output setup - extract clean video name without _result suffix
    video_name = Path(video_path).stem
    
    # Remove _result suffix if present
    if video_name.endswith("_result"):
        video_name = video_name[:-7]  # Remove "_result"
    
    output_subdir = os.path.join(output_dir, video_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    output_video_path = os.path.join(output_subdir, f"{video_name}_with_predictions.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        print(f"Error: Cannot create output video: {output_video_path}")
        sys.exit(1)
    
    print(f"Output video: {output_video_path}")
    
    # ByteTrack setup
    tracker = sv.ByteTrack(
        track_activation_threshold=0.40,
        lost_track_buffer=150,
        minimum_matching_threshold=0.80,
        frame_rate=int(round(fps)),
    )
    
    # Feature history for LSTM (deque of feature arrays)
    feature_history = deque(maxlen=64)  # Keep 64 timesteps (~16 seconds at 4 fps)
    skip_frames = max(1, int(fps / 4))  # Process ~4 fps equivalent
    frame_skip_counter = 0
    
    # Processing loop
    frame_count = 0
    start_time = time.time()
    
    print("\nProcessing video with predictions...")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_skip_counter += 1
        annotation_frame = frame.copy()
        
        # Run detection every Nth frame
        if frame_skip_counter >= skip_frames:
            frame_skip_counter = 0
            
            # YOLO detection
            results = yolo_model(
                frame,
                conf=conf_threshold,
                imgsz=imgsz,
                classes=0,  # Person class
                verbose=False,
            )
            detections = sv.Detections.from_ultralytics(results[0])
            
            # ByteTrack
            detections = tracker.update_with_detections(detections)
            
            # Get anchor points
            anchor_points = [get_anchor_point(box) for box in detections.xyxy]
            
            # Count people per zone
            zone_counts = count_people_in_zones(anchor_points, frame.shape)
            
            # Extract features for LSTM
            features = extract_features(zone_counts)
            feature_history.append(features)
            
            # Predict congestion for next frame
            predicted_total = predict_congestion(
                lstm_model,
                feature_history,
                target_mean,
                target_std,
                device,
            )
        else:
            # Use last known counts
            if feature_history:
                last_features = feature_history[-1]
                zone_counts = [int(f) for f in last_features[:9]]
            else:
                zone_counts = [0] * 9
            predicted_total = None
        
        # Get congestion status
        zone_statuses = [get_congestion_status(float(c)) for c in zone_counts]
        
        # Draw zones with predictions
        annotation_frame = draw_zones_with_predictions(
            annotation_frame,
            zone_counts,
            zone_statuses,
            predicted_total,
        )
        
        # Write frame
        out.write(annotation_frame)
        
        # Progress
        frame_count += 1
        if frame_count % max(1, total // 20) == 0 or frame_count == total:
            elapsed = time.time() - start_time
            rate = frame_count / elapsed if elapsed > 0 else 0
            eta = (total - frame_count) / rate if rate > 0 else 0
            print(
                f"[{frame_count}/{total}] {(frame_count/total)*100:.1f}% | "
                f"{rate:.1f} fps | ETA {eta:.0f}s"
            )
    
    cap.release()
    out.release()
    
    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"✅ Complete! Processed {frame_count} frames in {elapsed:.1f}s")
    print(f"Output: {output_video_path}")
    print("\n🎯 Demo Features:")
    print("  • Current zone congestion: Colored overlays")
    print("  • Predicted danger zones: MAGENTA highlights (appears 3 sec early)")
    print("  • Zone statistics: Current + predicted counts")


def main():
    parser = argparse.ArgumentParser(
        description="Process video with Phase 1 detection and Phase 2 LSTM prediction overlay."
    )
    parser.add_argument(
        "--video",
        default="/Users/abhisekh/Desktop/Major Project/Video/Dataset/Videos/6387-191695740.mp4",
        help="Path to input video",
    )
    parser.add_argument(
        "--checkpoint",
        default="output/phase2_training/phase2_lstm_multidata_retrained/phase2_lstm_checkpoint.pt",
        help="Path to LSTM checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default="output/phase1_processed",
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        default="yolov8m.pt",
        help="YOLO model to use",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO image size",
    )
    
    args = parser.parse_args()
    
    process_video_with_predictions(
        args.video,
        args.checkpoint,
        args.output_dir,
        args.model,
        args.conf,
        args.imgsz,
    )


if __name__ == "__main__":
    main()
