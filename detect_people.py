import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# Ensure local supervision module in src/ is accessible as fallback
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.exists(_src_dir) and _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    import supervision as sv
except ImportError:
    import src.supervision as sv

try:
    from tkinter import Tk, filedialog
except ImportError:
    Tk, filedialog = None, None

from ultralytics import YOLO



DEFAULT_MODEL_CANDIDATES = [
    "yolov8m.pt",
    "yolo11x.pt",
    "rtdetr-l.pt",
    "yolov8s.pt",
    "yolov8n.pt",
    "yolov8l.pt",
]

FLOW_FIELDNAMES = [
    "frame",
    "timestamp_sec",
    "tracker_id",
    "confidence",
    "zone_number",
    "congestion_status",
    "anchor_x",
    "anchor_y",
    "track_dx_per_frame",
    "track_dy_per_frame",
    "optical_flow_dx_per_frame",
    "optical_flow_dy_per_frame",
    "fused_dx_per_frame",
    "fused_dy_per_frame",
    "velocity_x_px_per_sec",
    "velocity_y_px_per_sec",
    "speed_px_per_sec",
    "direction_degrees",
]

DETECTION_FIELDNAMES = [
    "frame",
    "timestamp_sec",
    "tracker_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "anchor_x",
    "anchor_y",
    "width",
    "height",
    "area",
    "zone_number",
]

ZONE_FIELDNAMES = [
    "window_start_sec",
    "zone_number",
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
    "dominant_direction",
    "congestion_status",
]

TRANSITION_FIELDNAMES = [
    "frame",
    "timestamp_sec",
    "tracker_id",
    "from_zone",
    "to_zone",
]


@dataclass
class StableTrackState:
    stable_id: int
    raw_tracker_id: int
    last_frame: int
    box: tuple[float, float, float, float]
    anchor: tuple[float, float]


def resolve_default_model() -> str:
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return "yolov8m.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Person detection, tracking, and crowd-flow feature extraction."
    )
    parser.add_argument("video_path", nargs="?", help="Path to the input video.")
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="Optional output video filename. Saved inside the run folder.",
    )
    parser.add_argument(
        "grid_size",
        nargs="?",
        type=int,
        default=3,
        help="Grid size for zone-based analysis.",
    )
    parser.add_argument(
        "--model",
        default=resolve_default_model(),
        help="Model weights to use for person detection.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=0.0,
        help="Minimum bounding-box area in pixels. 0 disables area filtering.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO inference image size. Larger values help detect small people.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=500,
        help="Maximum number of detections kept per frame.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.50,
        help="NMS IOU threshold for detector post-processing.",
    )
    parser.add_argument(
        "--flow-scale",
        type=float,
        default=0.25,
        help="Resize factor used before computing dense optical flow.",
    )
    parser.add_argument(
        "--flow-interval",
        type=int,
        default=3,
        help="Compute optical flow every N frames and reuse the last result in between.",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.35,
        help="Exponential smoothing factor for tracked anchor points.",
    )
    parser.add_argument(
        "--track-activation-threshold",
        type=float,
        default=0.20,
        help="ByteTrack activation threshold.",
    )
    parser.add_argument(
        "--lost-track-buffer",
        type=int,
        default=40,
        help="ByteTrack lost-track buffer in frames.",
    )
    parser.add_argument(
        "--minimum-matching-threshold",
        type=float,
        default=0.85,
        help="ByteTrack matching threshold.",
    )
    parser.add_argument(
        "--use-slicer",
        action="store_true",
        help="Use tiled inference for small and corner people in high-resolution videos.",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=960,
        help="Tile size used when slicer mode is enabled.",
    )
    parser.add_argument(
        "--slice-overlap",
        type=int,
        default=160,
        help="Tile overlap used when slicer mode is enabled.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Append confidence values to output-video ID labels.",
    )
    parser.add_argument(
        "--stable-id-gap-seconds",
        type=float,
        default=1.5,
        help="Maximum gap used to reconnect a reappearing track to its previous ID.",
    )
    parser.add_argument(
        "--stable-id-distance-ratio",
        type=float,
        default=0.08,
        help="Maximum normalized anchor distance used for stable-ID reassignment.",
    )
    parser.add_argument(
        "--stable-id-min-iou",
        type=float,
        default=0.10,
        help="Minimum IOU hint used when reconnecting short occlusion gaps.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the live preview window and only save the final output.",
    )
    parser.add_argument(
        "--output",
        dest="flag_output_path",
        default=None,
        help="Optional output video filename alias.",
    )
    args = parser.parse_args()
    if args.flag_output_path and not args.output_path:
        args.output_path = args.flag_output_path
    return args


def draw_grid(
    frame: np.ndarray,
    zone_counts: list[int],
    zone_statuses: list[str],
    zone_tracker_ids: list[list[int]] | None = None,
    grid_size: int = 3,
) -> np.ndarray:
    h, w = frame.shape[:2]
    cell_w = w // grid_size
    cell_h = h // grid_size
    overlay = frame.copy()

    for row in range(grid_size):
        for col in range(grid_size):
            zone_id = row * grid_size + col
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = w if col == grid_size - 1 else (col + 1) * cell_w
            y2 = h if row == grid_size - 1 else (row + 1) * cell_h

            congestion = zone_statuses[zone_id]
            if congestion == "LOW":
                color = (0, 120, 0)
            elif congestion == "MEDIUM":
                color = (0, 180, 180)
            elif congestion == "HIGH":
                color = (0, 165, 255)
            else:
                color = (0, 0, 180)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    frame[:] = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)

    for i in range(1, grid_size):
        x = i * cell_w
        cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 2)

    for i in range(1, grid_size):
        y = i * cell_h
        cv2.line(frame, (0, y), (w, y), (255, 255, 255), 2)

    for row in range(grid_size):
        for col in range(grid_size):
            zone_index = row * grid_size + col
            x = col * cell_w + 10
            y = row * cell_h + 30
            cv2.putText(
                frame,
                f"Z{zone_index + 1}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"{zone_counts[zone_index]} | {zone_statuses[zone_index]}",
                (x, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            if zone_tracker_ids is not None:
                ids = zone_tracker_ids[zone_index]
                if ids:
                    id_text = ",".join(str(tracker_id) for tracker_id in ids)
                else:
                    id_text = "-"

                max_chars = max(8, int((cell_w - 20) / 12))
                id_chunks = [
                    id_text[i : i + max_chars] for i in range(0, len(id_text), max_chars)
                ]
                for line_index, chunk in enumerate(id_chunks[:3]):
                    prefix = "IDs:" if line_index == 0 else "    "
                    cv2.putText(
                        frame,
                        f"{prefix} {chunk}",
                        (x, y + 54 + (line_index * 22)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

    return frame


def get_anchor_point(box: np.ndarray) -> tuple[float, float]:
    x1, _y1, x2, y2 = box
    return (float((x1 + x2) / 2.0), float(y2))


def count_people_in_zones(
    anchor_points: list[tuple[float, float]],
    frame_shape: tuple[int, ...],
    grid_size: int = 3,
) -> list[int]:
    zone_counts = [0] * (grid_size * grid_size)
    for cx, cy in anchor_points:
        zone_id = get_zone_number(cx, cy, frame_shape, grid_size) - 1
        zone_counts[zone_id] += 1
    return zone_counts


def get_zone_number(
    cx: float, cy: float, frame_shape: tuple[int, ...], grid_size: int = 3
) -> int:
    h, w = frame_shape[:2]
    cell_w = max(1, w // grid_size)
    cell_h = max(1, h // grid_size)
    col = max(0, min(int(cx // cell_w), grid_size - 1))
    row = max(0, min(int(cy // cell_h), grid_size - 1))
    return row * grid_size + col + 1


def calculate_congestion_index(
    people_count: float,
    speed_px_per_sec: float = 65.0,
    zone_capacity: float = 15.0,
    free_flow_speed: float = 75.0,
    inflow_net: float = 0.0,
) -> tuple[float, str]:
    """Calculate Multi-Modal Congestion Hazard Index (CHI) for Transit Hubs.
    
    Combines:
    1. Density vs Capacity Ratio (55% weight)
    2. Velocity Drop / Stagnation Factor (30% weight)
    3. Inflow Pressure / Net Inrush (15% weight)
    """
    safe_cap = max(1.0, float(zone_capacity))
    density_ratio = min(2.0, float(people_count) / safe_cap)
    
    safe_free_speed = max(10.0, float(free_flow_speed))
    clamped_speed = max(0.0, min(float(speed_px_per_sec), safe_free_speed))
    velocity_stagnation = 1.0 - (clamped_speed / safe_free_speed)
    
    inflow_factor = min(1.0, max(0.0, float(inflow_net) / 5.0))
    
    chi = (0.55 * density_ratio) + (0.30 * velocity_stagnation) + (0.15 * inflow_factor)
    
    if chi < 0.40:
        status = "LOW"
    elif chi < 0.75:
        status = "MEDIUM"
    elif chi < 1.10:
        status = "HIGH"
    else:
        status = "CRITICAL"
        
    return round(chi, 3), status


def get_congestion_status(
    people_count: float,
    speed_px_per_sec: float = 65.0,
    zone_capacity: float = 15.0,
    free_flow_speed: float = 75.0,
) -> str:
    """Classify congestion status using calibrated multi-modal index."""
    _, status = calculate_congestion_index(
        people_count=people_count,
        speed_px_per_sec=speed_px_per_sec,
        zone_capacity=zone_capacity,
        free_flow_speed=free_flow_speed,
    )
    return status


def get_speed_color(speed_px_per_sec: float) -> tuple[int, int, int]:
    if speed_px_per_sec < 35:
        return (255, 0, 0)
    if speed_px_per_sec < 90:
        return (0, 255, 0)
    if speed_px_per_sec < 180:
        return (0, 255, 255)
    return (0, 0, 255)


def append_smoothed_point(
    history: deque[tuple[float, float]],
    point: tuple[float, float],
    alpha: float,
) -> tuple[float, float]:
    if history:
        prev_x, prev_y = history[-1]
        smoothed = (
            alpha * point[0] + (1.0 - alpha) * prev_x,
            alpha * point[1] + (1.0 - alpha) * prev_y,
        )
    else:
        smoothed = point
    history.append(smoothed)
    return smoothed


def compute_box_iou(box_a: np.ndarray, box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def resolve_stable_tracker_id(
    raw_tracker_id: int,
    box: np.ndarray,
    anchor_point: tuple[float, float],
    frame_index: int,
    frame_shape: tuple[int, ...],
    raw_to_stable: dict[int, int],
    stable_states: dict[int, StableTrackState],
    assigned_this_frame: set[int],
    next_stable_id: int,
    max_gap_frames: int,
    max_distance_ratio: float,
    min_iou: float,
) -> tuple[int, int]:
    existing_stable_id = raw_to_stable.get(raw_tracker_id)
    if existing_stable_id is not None and existing_stable_id not in assigned_this_frame:
        state = stable_states.get(existing_stable_id)
        if state is None or frame_index - state.last_frame <= max_gap_frames:
            assigned_this_frame.add(existing_stable_id)
            return existing_stable_id, next_stable_id

    frame_h, frame_w = frame_shape[:2]
    frame_diag = max(1.0, math.hypot(frame_w, frame_h))
    best_stable_id: int | None = None
    best_score = -1.0

    for stable_id, state in stable_states.items():
        if stable_id in assigned_this_frame:
            continue

        gap_frames = frame_index - state.last_frame
        if gap_frames <= 0 or gap_frames > max_gap_frames:
            continue

        iou = compute_box_iou(box, state.box)
        distance = math.hypot(
            anchor_point[0] - state.anchor[0],
            anchor_point[1] - state.anchor[1],
        )
        normalized_distance = distance / frame_diag

        current_area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
        previous_area = max(
            1.0, float((state.box[2] - state.box[0]) * (state.box[3] - state.box[1]))
        )
        area_similarity = min(current_area, previous_area) / max(current_area, previous_area)

        if iou < min_iou and normalized_distance > max_distance_ratio:
            continue

        score = (
            (iou * 0.60)
            + (max(0.0, 1.0 - (normalized_distance / max_distance_ratio)) * 0.25)
            + (area_similarity * 0.10)
            + (max(0.0, 1.0 - (gap_frames / max_gap_frames)) * 0.05)
        )
        if score > best_score:
            best_score = score
            best_stable_id = stable_id

    if best_stable_id is not None:
        raw_to_stable[raw_tracker_id] = best_stable_id
        assigned_this_frame.add(best_stable_id)
        return best_stable_id, next_stable_id

    stable_id = next_stable_id
    raw_to_stable[raw_tracker_id] = stable_id
    assigned_this_frame.add(stable_id)
    return stable_id, next_stable_id + 1


def prune_stable_track_states(
    stable_states: dict[int, StableTrackState],
    frame_index: int,
    max_gap_frames: int,
) -> None:
    stale_ids = [
        stable_id
        for stable_id, state in stable_states.items()
        if frame_index - state.last_frame > max_gap_frames
    ]
    for stable_id in stale_ids:
        stable_states.pop(stable_id, None)


def compute_track_motion(
    history: deque[tuple[float, float]], fps: float
) -> tuple[float, float, float, float, float, float]:
    if len(history) < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    start_x, start_y = history[0]
    end_x, end_y = history[-1]
    frame_delta = max(1, len(history) - 1)

    dx_per_frame = (end_x - start_x) / frame_delta
    dy_per_frame = (end_y - start_y) / frame_delta
    velocity_x = dx_per_frame * fps
    velocity_y = dy_per_frame * fps
    speed = math.hypot(velocity_x, velocity_y)
    angle = math.degrees(math.atan2(velocity_y, velocity_x)) if speed > 0 else 0.0

    return dx_per_frame, dy_per_frame, velocity_x, velocity_y, speed, angle


def compute_dense_flow(
    prev_gray_small: np.ndarray | None,
    gray_small: np.ndarray,
    frame_index: int,
    flow_interval: int,
    last_flow: np.ndarray | None,
) -> np.ndarray | None:
    if prev_gray_small is None:
        return None

    if frame_index % max(1, flow_interval) != 0:
        return last_flow

    return cv2.calcOpticalFlowFarneback(
        prev_gray_small,
        gray_small,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )


def extract_roi_flow(
    dense_flow: np.ndarray | None,
    box: np.ndarray,
    scale: float,
) -> tuple[float, float]:
    if dense_flow is None or scale <= 0:
        return 0.0, 0.0

    x1, y1, x2, y2 = box
    height = max(1.0, y2 - y1)

    roi_x1 = int(max(0, math.floor(x1 * scale)))
    roi_x2 = int(min(dense_flow.shape[1], math.ceil(x2 * scale)))
    roi_y1 = int(max(0, math.floor((y1 + 0.45 * height) * scale)))
    roi_y2 = int(min(dense_flow.shape[0], math.ceil(y2 * scale)))

    if roi_x2 - roi_x1 < 2 or roi_y2 - roi_y1 < 2:
        return 0.0, 0.0

    roi_flow = dense_flow[roi_y1:roi_y2, roi_x1:roi_x2]
    dx = float(np.median(roi_flow[..., 0])) / scale
    dy = float(np.median(roi_flow[..., 1])) / scale
    return dx, dy


def fuse_motion_vectors(
    track_dx: float,
    track_dy: float,
    flow_dx: float,
    flow_dy: float,
    history_length: int,
) -> tuple[float, float]:
    if history_length < 3:
        return flow_dx, flow_dy

    track_weight = 0.75 if history_length >= 5 else 0.60
    flow_weight = 1.0 - track_weight
    return (
        track_weight * track_dx + flow_weight * flow_dx,
        track_weight * track_dy + flow_weight * flow_dy,
    )


def draw_motion_arrow(
    frame: np.ndarray,
    point: tuple[float, float],
    dx_per_frame: float,
    dy_per_frame: float,
    speed_px_per_sec: float,
    scale: float = 6.0,
    thickness: int = 2,
    max_length: float = 85.0,
) -> None:
    vector_x = dx_per_frame * scale
    vector_y = dy_per_frame * scale
    magnitude = math.hypot(vector_x, vector_y)
    if magnitude < 1.0:
        return

    if magnitude > max_length:
        ratio = max_length / magnitude
        vector_x *= ratio
        vector_y *= ratio

    start = (int(point[0]), int(point[1]))
    end = (int(point[0] + vector_x), int(point[1] + vector_y))
    cv2.arrowedLine(
        frame,
        start,
        end,
        get_speed_color(speed_px_per_sec),
        thickness,
        tipLength=0.25,
    )


def classify_direction(dx: float, dy: float) -> str:
    magnitude = math.hypot(dx, dy)
    if magnitude < 0.3:
        return "STILL"

    angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
    directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    index = int(((angle + 22.5) % 360.0) // 45.0)
    return directions[index]


def update_zone_window(
    zone_windows: dict[tuple[int, int], dict[str, Any]],
    window_start_sec: int,
    zone_number: int,
) -> dict[str, Any]:
    key = (window_start_sec, zone_number)
    if key not in zone_windows:
        zone_windows[key] = {
            "window_start_sec": window_start_sec,
            "zone_number": zone_number,
            "count_sum": 0.0,
            "count_samples": 0,
            "max_people_count": 0,
            "track_ids": set(),
            "speed_sum": 0.0,
            "speed_samples": 0,
            "fused_dx_sum": 0.0,
            "fused_dy_sum": 0.0,
            "flow_dx_sum": 0.0,
            "flow_dy_sum": 0.0,
            "motion_samples": 0,
            "inflow_count": 0,
            "outflow_count": 0,
        }
    return zone_windows[key]


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_detector(
    model: YOLO,
    confidence_threshold: float,
    imgsz: int,
    max_det: int,
    nms_iou: float,
    use_slicer: bool,
    slice_size: int,
    slice_overlap: int,
) -> Any:
    def detect(image: np.ndarray) -> sv.Detections:
        results = model(
            image,
            classes=[0],
            conf=confidence_threshold,
            imgsz=imgsz,
            max_det=max_det,
            iou=nms_iou,
            verbose=False,
        )
        return sv.Detections.from_ultralytics(results[0])

    if not use_slicer:
        return detect

    return sv.InferenceSlicer(
        callback=detect,
        slice_wh=(slice_size, slice_size),
        overlap_wh=(slice_overlap, slice_overlap),
        iou_threshold=nms_iou,
        thread_workers=1,
    )


def process_video(
    video_path: str,
    output_path: str | None = None,
    grid_size: int = 3,
    model_path: str | None = None,
    confidence_threshold: float = 0.30,
    min_box_area: float = 0.0,
    imgsz: int = 1280,
    max_det: int = 500,
    nms_iou: float = 0.50,
    flow_scale: float = 0.25,
    flow_interval: int = 3,
    smoothing_alpha: float = 0.35,
    track_activation_threshold: float = 0.20,
    lost_track_buffer: int = 40,
    minimum_matching_threshold: float = 0.85,
    use_slicer: bool = False,
    slice_size: int = 960,
    slice_overlap: int = 160,
    show_labels: bool = False,
    stable_id_gap_seconds: float = 1.5,
    stable_id_distance_ratio: float = 0.08,
    stable_id_min_iou: float = 0.10,
    no_preview: bool = False,
    frame_callback: Any | None = None,
) -> None:
    model_path = model_path or resolve_default_model()
    print(f"Loading: {video_path}")
    print(f"Model: {model_path} | Conf: {confidence_threshold:.2f}")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    effective_min_box_area = max(0.0, min_box_area)

    tracker = sv.ByteTrack(
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=lost_track_buffer,
        minimum_matching_threshold=minimum_matching_threshold,
        frame_rate=int(round(fps)),
    )
    detector = build_detector(
        model=model,
        confidence_threshold=confidence_threshold,
        imgsz=imgsz,
        max_det=max_det,
        nms_iou=nms_iou,
        use_slicer=use_slicer,
        slice_size=slice_size,
        slice_overlap=slice_overlap,
    )

    input_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_root = os.path.join(os.getcwd(), "output", "phase1_processed")
    output_dir = os.path.join(output_root, input_filename)
    os.makedirs(output_dir, exist_ok=True)

    final_output_name = os.path.basename(output_path) if output_path else f"{input_filename}_result.mp4"
    final_output_path = os.path.join(output_dir, final_output_name)

    print(f"Resolution: {w}x{h} | FPS: {fps:.2f} | Frames: {total}")
    print(
        f"Grid: {grid_size}x{grid_size} | Min box area: {effective_min_box_area:.0f} | "
        f"imgsz: {imgsz} | max_det: {max_det}"
    )
    print(
        f"Optical flow scale: {flow_scale:.2f} | interval: {max(1, flow_interval)} | "
        f"slicer: {'ON' if use_slicer else 'OFF'}"
    )
    print(f"Preview: {'OFF' if no_preview else 'ON'}")
    print(f"Output dir: {output_dir}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(final_output_path, fourcc, fps, (w, h))

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator()

    history_length = max(6, min(18, int(round(fps * 0.5))))
    smoothed_history = defaultdict(lambda: deque(maxlen=history_length))
    people_history: list[int] = []
    frame_times: list[float] = []
    last_zone_by_tracker: dict[int, int] = {}
    raw_to_stable_id: dict[int, int] = {}
    stable_states: dict[int, StableTrackState] = {}
    next_stable_id = 1
    stable_id_gap_frames = max(1, int(round(stable_id_gap_seconds * fps)))

    detection_rows: list[dict[str, Any]] = []
    flow_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    zone_windows: dict[tuple[int, int], dict[str, Any]] = {}

    prev_gray_small: np.ndarray | None = None
    last_dense_flow: np.ndarray | None = None
    frame_count = 0
    paused = False

    print("Processing...")
    print("  Press 'p' to pause/resume | 'q' to quit")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector(frame)

        if len(detections) > 0 and effective_min_box_area > 0:
            areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (
                detections.xyxy[:, 3] - detections.xyxy[:, 1]
            )
            detections = detections[areas >= effective_min_box_area]

        people = tracker.update_with_detections(detections)

        gray_small: np.ndarray | None = None
        dense_flow: np.ndarray | None = last_dense_flow
        if len(people) > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(
                gray,
                None,
                fx=flow_scale,
                fy=flow_scale,
                interpolation=cv2.INTER_AREA,
            )
            dense_flow = compute_dense_flow(
                prev_gray_small=prev_gray_small,
                gray_small=gray_small,
                frame_index=frame_count,
                flow_interval=flow_interval,
                last_flow=last_dense_flow,
            )

        anchor_points = [get_anchor_point(box) for box in people.xyxy]
        zone_counts = count_people_in_zones(anchor_points, frame.shape, grid_size)
        zone_statuses = [get_congestion_status(count) for count in zone_counts]
        zone_tracker_ids: list[list[int]] = [[] for _ in range(grid_size * grid_size)]

        window_start_sec = int(frame_count / fps)
        for zone_number, zone_count in enumerate(zone_counts, start=1):
            zone_window = update_zone_window(zone_windows, window_start_sec, zone_number)
            zone_window["count_sum"] += zone_count
            zone_window["count_samples"] += 1
            zone_window["max_people_count"] = max(zone_window["max_people_count"], zone_count)

        labels: list[str] = []
        out_frame = box_annotator.annotate(scene=frame.copy(), detections=people)
        assigned_stable_ids: set[int] = set()

        for box, tracker_id, confidence in zip(
            people.xyxy,
            people.tracker_id,
            people.confidence,
        ):
            if tracker_id is None:
                continue

            raw_tracker_id = int(tracker_id)
            anchor_x, anchor_y = get_anchor_point(box)
            tracker_id_int, next_stable_id = resolve_stable_tracker_id(
                raw_tracker_id=raw_tracker_id,
                box=box,
                anchor_point=(anchor_x, anchor_y),
                frame_index=frame_count,
                frame_shape=frame.shape,
                raw_to_stable=raw_to_stable_id,
                stable_states=stable_states,
                assigned_this_frame=assigned_stable_ids,
                next_stable_id=next_stable_id,
                max_gap_frames=stable_id_gap_frames,
                max_distance_ratio=stable_id_distance_ratio,
                min_iou=stable_id_min_iou,
            )
            stable_states[tracker_id_int] = StableTrackState(
                stable_id=tracker_id_int,
                raw_tracker_id=raw_tracker_id,
                last_frame=frame_count,
                box=tuple(float(v) for v in box),
                anchor=(anchor_x, anchor_y),
            )
            zone_number = get_zone_number(anchor_x, anchor_y, frame.shape, grid_size)
            congestion = zone_statuses[zone_number - 1]
            zone_tracker_ids[zone_number - 1].append(tracker_id_int)

            smoothed_point = append_smoothed_point(
                smoothed_history[tracker_id_int],
                (anchor_x, anchor_y),
                smoothing_alpha,
            )
            (
                track_dx_per_frame,
                track_dy_per_frame,
                _track_vx,
                _track_vy,
                _track_speed,
                _track_angle,
            ) = compute_track_motion(smoothed_history[tracker_id_int], fps)

            optical_flow_dx, optical_flow_dy = extract_roi_flow(dense_flow, box, flow_scale)
            fused_dx, fused_dy = fuse_motion_vectors(
                track_dx=track_dx_per_frame,
                track_dy=track_dy_per_frame,
                flow_dx=optical_flow_dx,
                flow_dy=optical_flow_dy,
                history_length=len(smoothed_history[tracker_id_int]),
            )

            velocity_x = fused_dx * fps
            velocity_y = fused_dy * fps
            speed = math.hypot(velocity_x, velocity_y)
            direction_degrees = (
                math.degrees(math.atan2(velocity_y, velocity_x)) if speed > 0 else 0.0
            )

            label_text = f"ID {tracker_id_int}"
            if show_labels:
                label_text = f"{label_text} {confidence:.2f}"
            labels.append(label_text)
            cv2.circle(out_frame, (int(smoothed_point[0]), int(smoothed_point[1])), 5, (0, 0, 255), -1)
            draw_motion_arrow(
                frame=out_frame,
                point=smoothed_point,
                dx_per_frame=fused_dx,
                dy_per_frame=fused_dy,
                speed_px_per_sec=speed,
            )

            detection_rows.append(
                {
                    "frame": frame_count,
                    "timestamp_sec": round(frame_count / fps, 3),
                    "tracker_id": tracker_id_int,
                    "confidence": round(float(confidence), 4),
                    "x1": round(float(box[0]), 2),
                    "y1": round(float(box[1]), 2),
                    "x2": round(float(box[2]), 2),
                    "y2": round(float(box[3]), 2),
                    "anchor_x": round(anchor_x, 2),
                    "anchor_y": round(anchor_y, 2),
                    "width": round(float(box[2] - box[0]), 2),
                    "height": round(float(box[3] - box[1]), 2),
                    "area": round(float((box[2] - box[0]) * (box[3] - box[1])), 2),
                    "zone_number": zone_number,
                }
            )

            flow_rows.append(
                {
                    "frame": frame_count,
                    "timestamp_sec": round(frame_count / fps, 3),
                    "tracker_id": tracker_id_int,
                    "confidence": round(float(confidence), 4),
                    "zone_number": zone_number,
                    "congestion_status": congestion,
                    "anchor_x": round(anchor_x, 2),
                    "anchor_y": round(anchor_y, 2),
                    "track_dx_per_frame": round(track_dx_per_frame, 4),
                    "track_dy_per_frame": round(track_dy_per_frame, 4),
                    "optical_flow_dx_per_frame": round(optical_flow_dx, 4),
                    "optical_flow_dy_per_frame": round(optical_flow_dy, 4),
                    "fused_dx_per_frame": round(fused_dx, 4),
                    "fused_dy_per_frame": round(fused_dy, 4),
                    "velocity_x_px_per_sec": round(velocity_x, 2),
                    "velocity_y_px_per_sec": round(velocity_y, 2),
                    "speed_px_per_sec": round(speed, 2),
                    "direction_degrees": round(direction_degrees, 2),
                }
            )

            zone_window = update_zone_window(zone_windows, window_start_sec, zone_number)
            zone_window["track_ids"].add(tracker_id_int)
            zone_window["speed_sum"] += speed
            zone_window["speed_samples"] += 1
            zone_window["fused_dx_sum"] += fused_dx
            zone_window["fused_dy_sum"] += fused_dy
            zone_window["flow_dx_sum"] += optical_flow_dx
            zone_window["flow_dy_sum"] += optical_flow_dy
            zone_window["motion_samples"] += 1

            previous_zone = last_zone_by_tracker.get(tracker_id_int)
            if previous_zone is not None and previous_zone != zone_number:
                update_zone_window(zone_windows, window_start_sec, previous_zone)["outflow_count"] += 1
                zone_window["inflow_count"] += 1
                transition_rows.append(
                    {
                        "frame": frame_count,
                        "timestamp_sec": round(frame_count / fps, 3),
                        "tracker_id": tracker_id_int,
                        "from_zone": previous_zone,
                        "to_zone": zone_number,
                    }
                )
            last_zone_by_tracker[tracker_id_int] = zone_number

        prune_stable_track_states(
            stable_states=stable_states,
            frame_index=frame_count,
            max_gap_frames=stable_id_gap_frames,
        )

        preview_frame = out_frame.copy()
        preview_frame = draw_grid(
            preview_frame, zone_counts, zone_statuses, grid_size=grid_size
        )
        cv2.putText(
            preview_frame,
            f"[{frame_count}/{total}] {(frame_count / total) * 100:.1f}%" if total > 0 else f"[{frame_count}]",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            preview_frame,
            f"People: {len(people)}",
            (10, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            preview_frame,
            f"Model: {os.path.basename(model_path)}",
            (10, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        if paused:
            cv2.putText(
                preview_frame,
                "PAUSED (Press P to Resume)",
                (10, preview_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        if labels:
            out_frame = label_annotator.annotate(
                scene=out_frame, detections=people, labels=labels
            )
        zone_tracker_ids = [sorted(set(ids)) for ids in zone_tracker_ids]
        out_frame = draw_grid(
            out_frame,
            zone_counts,
            zone_statuses,
            zone_tracker_ids=zone_tracker_ids,
            grid_size=grid_size,
        )

        cv2.putText(
            out_frame,
            f"[{frame_count}/{total}] {(frame_count / total) * 100:.1f}%" if total > 0 else f"[{frame_count}]",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            out_frame,
            f"People: {len(people)}",
            (10, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            out_frame,
            f"Model: {os.path.basename(model_path)}",
            (10, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        if paused:
            cv2.putText(
                out_frame,
                "PAUSED (Press P to Resume)",
                (10, out_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        if frame_callback is not None:
            try:
                frame_callback(preview_frame, frame_count, total)
            except Exception:
                pass

        if not no_preview:
            try:
                cv2.imshow("Live", preview_frame)
            except (cv2.error, Exception):
                no_preview = True

        if not paused:
            out.write(out_frame)
            frame_count += 1
            people_history.append(len(people))
            frame_times.append(time.time() - t0)
            if len(frame_times) > 30:
                frame_times.pop(0)

            if gray_small is not None:
                prev_gray_small = gray_small
            if dense_flow is not None:
                last_dense_flow = dense_flow

            if frame_count % max(1, total // 10) == 0 or frame_count % 100 == 0:
                avg_people = sum(people_history[-30:]) / min(30, len(people_history))
                fps_now = len(frame_times) / sum(frame_times) if frame_times else 0.0
                print(
                    f"  [{frame_count}/{total}] "
                    f"{(frame_count / total) * 100:.1f}% | "
                    f"Avg people: {avg_people:.1f} | FPS: {fps_now:.1f}"
                )
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        if not no_preview:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q") or key == 27:
                print("Stopped")
                break
            if key == ord("p") or key == ord("P"):
                paused = not paused
                if paused:
                    print("  PAUSED - Press 'p' to resume")
                else:
                    print("  RESUMED")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    zone_rows: list[dict[str, Any]] = []
    for key in sorted(zone_windows):
        window = zone_windows[key]
        count_samples = max(1, window["count_samples"])
        motion_samples = max(1, window["motion_samples"])
        speed_samples = max(1, window["speed_samples"])

        avg_people_count = window["count_sum"] / count_samples
        mean_flow_dx = window["fused_dx_sum"] / motion_samples
        mean_flow_dy = window["fused_dy_sum"] / motion_samples

        zone_rows.append(
            {
                "window_start_sec": window["window_start_sec"],
                "zone_number": window["zone_number"],
                "avg_people_count": round(avg_people_count, 3),
                "max_people_count": window["max_people_count"],
                "unique_track_ids": len(window["track_ids"]),
                "avg_speed_px_per_sec": round(window["speed_sum"] / speed_samples, 3),
                "mean_flow_dx_per_frame": round(mean_flow_dx, 4),
                "mean_flow_dy_per_frame": round(mean_flow_dy, 4),
                "mean_optical_flow_dx_per_frame": round(
                    window["flow_dx_sum"] / motion_samples, 4
                ),
                "mean_optical_flow_dy_per_frame": round(
                    window["flow_dy_sum"] / motion_samples, 4
                ),
                "inflow_count": window["inflow_count"],
                "outflow_count": window["outflow_count"],
                "dominant_direction": classify_direction(mean_flow_dx, mean_flow_dy),
                "congestion_status": get_congestion_status(avg_people_count),
            }
        )

    flow_csv_path = os.path.join(output_dir, "flow_vectors.csv")
    detections_csv_path = os.path.join(output_dir, "tracked_detections.csv")
    zone_csv_path = os.path.join(output_dir, "zone_time_series.csv")
    transition_csv_path = os.path.join(output_dir, "zone_transitions.csv")
    metadata_path = os.path.join(output_dir, "run_metadata.json")

    write_csv(flow_csv_path, FLOW_FIELDNAMES, flow_rows)
    write_csv(detections_csv_path, DETECTION_FIELDNAMES, detection_rows)
    write_csv(zone_csv_path, ZONE_FIELDNAMES, zone_rows)
    write_csv(transition_csv_path, TRANSITION_FIELDNAMES, transition_rows)

    metadata = {
        "video_path": video_path,
        "output_video_path": final_output_path,
        "model_path": model_path,
        "confidence_threshold": confidence_threshold,
        "grid_size": grid_size,
        "fps": fps,
        "frame_width": w,
        "frame_height": h,
        "total_frames_processed": frame_count,
        "min_box_area": effective_min_box_area,
        "imgsz": imgsz,
        "max_det": max_det,
        "nms_iou": nms_iou,
        "flow_scale": flow_scale,
        "flow_interval": max(1, flow_interval),
        "smoothing_alpha": smoothing_alpha,
        "track_activation_threshold": track_activation_threshold,
        "lost_track_buffer": lost_track_buffer,
        "minimum_matching_threshold": minimum_matching_threshold,
        "use_slicer": use_slicer,
        "slice_size": slice_size,
        "slice_overlap": slice_overlap,
        "show_labels": show_labels,
        "stable_id_gap_seconds": stable_id_gap_seconds,
        "stable_id_distance_ratio": stable_id_distance_ratio,
        "stable_id_min_iou": stable_id_min_iou,
        "no_preview": no_preview,
    }
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved video: {final_output_path}")
    print(f"Saved detections: {detections_csv_path}")
    print(f"Saved flow vectors: {flow_csv_path}")
    print(f"Saved zone features: {zone_csv_path}")
    print(f"Saved transitions: {transition_csv_path}")
    print(f"Saved metadata: {metadata_path}")

    if people_history:
        avg_people = sum(people_history) / len(people_history)
        max_people = max(people_history)
        print(
            f"\nDone | Avg people/frame: {avg_people:.1f} | "
            f"Max people/frame: {max_people}"
        )
    else:
        print("\nDone")


def prompt_for_video() -> tuple[str, str]:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected_path = filedialog.askopenfilename(
        title="Select video",
        filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("All", "*.*")],
    )
    if not selected_path:
        print("No file selected")
        sys.exit(1)

    input_name = os.path.splitext(os.path.basename(selected_path))[0]
    suggested_output = f"{input_name}_result.mp4"
    print(f"Selected: {selected_path}")
    print(f"Output: {suggested_output}")
    return selected_path, suggested_output


if __name__ == "__main__":
    args = parse_args()
    video_path = args.video_path
    output_path = args.output_path

    if not video_path:
        video_path, suggested_output = prompt_for_video()
        if output_path is None:
            output_path = suggested_output

    process_video(
        video_path=video_path,
        output_path=output_path,
        grid_size=args.grid_size,
        model_path=args.model,
        confidence_threshold=args.conf,
        min_box_area=args.min_box_area,
        imgsz=args.imgsz,
        max_det=args.max_det,
        nms_iou=args.nms_iou,
        flow_scale=args.flow_scale,
        flow_interval=args.flow_interval,
        smoothing_alpha=args.smoothing_alpha,
        track_activation_threshold=args.track_activation_threshold,
        lost_track_buffer=args.lost_track_buffer,
        minimum_matching_threshold=args.minimum_matching_threshold,
        use_slicer=args.use_slicer,
        slice_size=args.slice_size,
        slice_overlap=args.slice_overlap,
        show_labels=args.show_labels,
        stable_id_gap_seconds=args.stable_id_gap_seconds,
        stable_id_distance_ratio=args.stable_id_distance_ratio,
        stable_id_min_iou=args.stable_id_min_iou,
        no_preview=args.no_preview,
    )
    
    # 🎯 AUTO-GENERATE PREDICTIONS AFTER DETECTION
    import subprocess
    print("\n" + "="*70)
    print("🎬 AUTO-GENERATING PREDICTION OVERLAY VIDEO...")
    print("=" * 70)
    
    try:
        checkpoint_path = "output/phase2_training/phase2_lstm_multidata_retrained/phase2_lstm_checkpoint.pt"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = "output/phase2_training/phase2_lstm/phase2_lstm_checkpoint.pt"

        pred_cmd = [
            "python3",
            "phase1_with_predictions.py",
            "--video", video_path,
            "--checkpoint", checkpoint_path,
            "--output-dir", "output/phase1_processed",
            "--model", args.model,
            "--conf", str(args.conf),
            "--imgsz", str(args.imgsz),
        ]
        
        result = subprocess.run(pred_cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print("✅ Prediction overlay video generated successfully!")
        else:
            err_msg = result.stderr.strip() or result.stdout.strip()
            print(f"⚠️  Could not generate prediction video (non-critical). Error:\n{err_msg}")
    except Exception as e:
        print(f"⚠️  Prediction generation skipped: {e}")
    
    print("=" * 70)

