import sys
import os
import cv2
import time
import csv
import math
import supervision as sv
from ultralytics import YOLO
from tkinter import Tk, filedialog
from collections import defaultdict, deque


def draw_grid(frame, grid_size=3):
    h, w = frame.shape[:2]
    cell_w = w // grid_size
    cell_h = h // grid_size
    
    for i in range(1, grid_size):
        x = i * cell_w
        cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 2)
    
    for i in range(1, grid_size):
        y = i * cell_h
        cv2.line(frame, (0, y), (w, y), (255, 255, 255), 2)
    
    for row in range(grid_size):
        for col in range(grid_size):
            x = col * cell_w + 10
            y = row * cell_h + 30
            zone_id = row * grid_size + col + 1
            cv2.putText(
                frame,
                f"Z{zone_id}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3
            )
    
    return frame


def count_people_in_zones(detections, frame_shape, grid_size=3):
    h, w = frame_shape[:2]
    cell_w = w // grid_size
    cell_h = h // grid_size
    zone_counts = [0] * (grid_size * grid_size)
    
    if len(detections) == 0:
        return zone_counts
    
    for box in detections.xyxy:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        col = int(center_x // cell_w)
        row = int(center_y // cell_h)
        
        col = max(0, min(col, grid_size - 1))
        row = max(0, min(row, grid_size - 1))
        
        zone_id = row * grid_size + col
        zone_counts[zone_id] += 1
    
    return zone_counts


def get_zone_number(cx, cy, frame_shape, grid_size=3):
    h, w = frame_shape[:2]
    cell_w = w // grid_size
    cell_h = h // grid_size
    col = max(0, min(int(cx // cell_w), grid_size - 1))
    row = max(0, min(int(cy // cell_h), grid_size - 1))
    return row * grid_size + col + 1


def get_speed_color(speed):
    if speed < 3:
        return (255, 0, 0)
    elif speed < 8:
        return (0, 255, 0)
    elif speed < 15:
        return (0, 255, 255)
    else:
        return (0, 0, 255)


def draw_movement_arrow(frame, cx, cy, history, scale=2.5, thickness=2):
    if len(history) < 4:
        return None, None
    
    history_list = list(history)
    
    start_x = int((history_list[-2][0] + history_list[-1][0]) / 2)
    start_y = int((history_list[-2][1] + history_list[-1][1]) / 2)
    
    current_x, current_y = int(cx), int(cy)
    
    old_x = int((history_list[0][0] + history_list[1][0]) / 2)
    old_y = int((history_list[0][1] + history_list[1][1]) / 2)
    
    dx = (current_x - old_x) * scale
    dy = (current_y - old_y) * scale
    
    end_x = int(current_x + dx)
    end_y = int(current_y + dy)
    
    speed = math.sqrt((current_x - old_x) ** 2 + (current_y - old_y) ** 2)
    color = get_speed_color(speed)
    
    if (current_x - old_x) != 0:
        angle_deg = math.degrees(math.atan2(current_y - old_y, current_x - old_x))
    else:
        angle_deg = 0
    
    cv2.arrowedLine(frame, (current_x, current_y), (end_x, end_y), color, thickness, tipLength=0.3)
    
    return speed, angle_deg


def get_congestion_status(people_count):
    if people_count < 5:
        return "LOW"
    elif people_count < 15:
        return "MEDIUM"
    elif people_count < 30:
        return "HIGH"
    else:
        return "CRITICAL"


def process_video(video_path, output_path="output.mp4", grid_size=3):
    
    print(f"Loading: {video_path}")
    model = YOLO("yolov8s.pt")
    model.conf = 0.35
    tracker = sv.ByteTrack()
    
    input_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.getcwd(), input_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    final_output_path = os.path.join(output_dir, f"{input_filename}_result.mp4")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Resolution: {w}x{h} | FPS: {fps} | Frames: {total}")
    print(f"Grid: {grid_size}x{grid_size} | Output dir: {output_dir}")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(final_output_path, fourcc, fps, (w, h))
    
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator()
    
    track_history = defaultdict(lambda: deque(maxlen=8))
    csv_data = []
    
    frame_count = 0
    people_history = []
    frame_times = []
    paused = False
    
    print("Processing...")
    print("  Press 'p' to pause/resume | 'q' to quit")
    
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        people = detections[detections.class_id == 0]
        
        people = tracker.update_with_detections(people)
        
        labels = [f"ID {int(tid)} {c:.2f}" for tid, c in zip(people.tracker_id, people.confidence)]
        
        out_frame = box_annotator.annotate(scene=frame.copy(), detections=people)
        out_frame = label_annotator.annotate(scene=out_frame, detections=people, labels=labels)
        
        out_frame = draw_grid(out_frame, grid_size=grid_size)
        zones = count_people_in_zones(people, out_frame.shape, grid_size)
        
        for box, tracker_id in zip(people.xyxy, people.tracker_id):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = y2
            
            track_history[int(tracker_id)].append((cx, cy))
            
            body_center_x = int(cx)
            body_center_y = int((y1 + y2) / 2)
            cv2.circle(out_frame, (body_center_x, body_center_y), 5, (0, 0, 255), -1)
            
            speed, angle = draw_movement_arrow(out_frame, cx, int(cy), track_history[int(tracker_id)])
            
            zone_num = get_zone_number(cx, cy, out_frame.shape, grid_size)
            congestion = get_congestion_status(zones[zone_num - 1])
            
            if speed is not None:
                hist_list = list(track_history[int(tracker_id)])
                dx = int((hist_list[-1][0] - hist_list[0][0]) * 2.5)
                dy = int((hist_list[-1][1] - hist_list[0][1]) * 2.5)
                csv_data.append({
                    'frame': frame_count,
                    'tracker_id': int(tracker_id),
                    'cx': int(cx),
                    'cy': int(cy),
                    'arrow_dx': dx,
                    'arrow_dy': dy,
                    'speed': round(speed, 2),
                    'direction_degrees': round(angle, 1),
                    'zone_number': zone_num,
                    'congestion_status': congestion
                })
        
        cv2.putText(out_frame, "Zones:", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        for i, cnt in enumerate(zones):
            y = 170 + (i * 35)
            cv2.putText(out_frame, f"  Z{i+1}: {cnt}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        pct = (frame_count / total) * 100 if total > 0 else 0
        cv2.putText(out_frame, f"[{frame_count}/{total}] {pct:.1f}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(out_frame, f"People: {len(people)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        if paused:
            cv2.putText(out_frame, "PAUSED (Press P to Resume)", (10, out_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        cv2.imshow("Live", out_frame)
        
        if not paused:
            out.write(out_frame)
            frame_count += 1
            people_history.append(len(people))
            frame_times.append(time.time() - t0)
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            if frame_count % max(1, total // 10) == 0 or frame_count % 100 == 0:
                avg_people = sum(people_history[-30:]) / min(30, len(people_history))
                fps_now = len(frame_times) / sum(frame_times) if frame_times else 0
                print(f"  [{frame_count}/{total}] {pct:.1f}% | Avg people: {avg_people:.1f} | FPS: {fps_now:.1f}")
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            print("Stopped")
            break
        if key == ord('p') or key == ord('P'):
            paused = not paused
            if paused:
                print("  ⏸ PAUSED - Press 'p' to resume")
            else:
                print("  ▶ RESUMED")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    csv_path = os.path.join(output_dir, "flow_vectors.csv")
    if csv_data:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame', 'tracker_id', 'cx', 'cy', 'arrow_dx', 'arrow_dy', 'speed', 'direction_degrees', 'zone_number', 'congestion_status'])
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"Saved flow vectors: {csv_path}")
    
    if people_history:
        avg = sum(people_history) / len(people_history)
        mx = max(people_history)
        print(f"\nDone | Output: {final_output_path} | Avg people/frame: {avg:.1f} | Max: {mx}")
    else:
        print(f"\nDone | Output: {final_output_path}")


if __name__ == "__main__":
    path = None
    out_path = "output.mp4"
    gs = 3
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"
        gs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    else:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("All", "*.*")]
        )
        
        if not path:
            print("No file selected")
            sys.exit(1)
        
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = f"{name}_result.mp4"
        
        print(f"Selected: {path}")
        print(f"Output: {out_path}")
    
    process_video(path, out_path, gs)
