import sys
import os
import cv2
import time
import supervision as sv
from ultralytics import YOLO
from tkinter import Tk, filedialog


def draw_grid(frame, grid_size=3):
    h, w = frame.shape[:2]
    cell_w = w // grid_size
    cell_h = h // grid_size
    
    for i in range(1, grid_size):
        x = i * cell_w
        cv2.line(frame, (x, 0), (x, h), (100, 100, 100), 2)
    
    for i in range(1, grid_size):
        y = i * cell_h
        cv2.line(frame, (0, y), (w, y), (100, 100, 100), 2)
    
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
                0.6,
                (200, 200, 200),
                2
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
        
        col = min(col, grid_size - 1)
        row = min(row, grid_size - 1)
        
        zone_id = row * grid_size + col
        zone_counts[zone_id] += 1
    
    return zone_counts


def process_video(video_path, output_path="output.mp4", grid_size=3):
    
    print(f"Loading: {video_path}")
    model = YOLO("yolov8n.pt")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Resolution: {w}x{h} | FPS: {fps} | Frames: {total}")
    print(f"Grid: {grid_size}x{grid_size} | Output: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator()
    
    frame_count = 0
    people_history = []
    frame_times = []
    
    print("Processing...")
    
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        people = detections[detections.class_id == 0]
        
        labels = [f"Person {c:.2f}" for c in people.confidence]
        
        out_frame = box_annotator.annotate(scene=frame.copy(), detections=people)
        out_frame = label_annotator.annotate(scene=out_frame, detections=people, labels=labels)
        
        out_frame = draw_grid(out_frame, grid_size=grid_size)
        zones = count_people_in_zones(people, out_frame.shape, grid_size)
        
        cv2.putText(out_frame, "Zones:", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        
        for i, cnt in enumerate(zones):
            y = 140 + (i * 25)
            cv2.putText(out_frame, f"  Z{i+1}: {cnt}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        
        pct = (frame_count / total) * 100 if total > 0 else 0
        cv2.putText(out_frame, f"[{frame_count}/{total}] {pct:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(out_frame, f"People: {len(people)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Live", out_frame)
        out.write(out_frame)
        
        frame_count += 1
        people_history.append(len(people))
        
        frame_times.append(time.time() - t0)
        if len(frame_times) > 30:
            frame_times.pop(0)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            print("Stopped")
            break
        
        if frame_count % max(1, total // 10) == 0 or frame_count % 100 == 0:
            avg_people = sum(people_history[-30:]) / min(30, len(people_history))
            fps_now = len(frame_times) / sum(frame_times) if frame_times else 0
            print(f"  [{frame_count}/{total}] {pct:.1f}% | Avg people: {avg_people:.1f} | FPS: {fps_now:.1f}")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    if people_history:
        avg = sum(people_history) / len(people_history)
        mx = max(people_history)
        print(f"\nDone | Output: {output_path} | Avg people/frame: {avg:.1f} | Max: {mx}")
    else:
        print(f"\nDone | Output: {output_path}")


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
