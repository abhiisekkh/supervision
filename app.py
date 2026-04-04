import os
import threading
import cv2
import time
import supervision as sv
from ultralytics import YOLO
from flask import Flask, render_template, request, Response, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = YOLO("yolov8n.pt")
processing_state = {
    'running': False,
    'frame_count': 0,
    'total_frames': 0,
    'fps': 0,
    'people_count': 0,
    'zone_counts': [],
    'progress': 0,
    'frame': None
}
processing_lock = threading.Lock()


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
            cv2.putText(frame, f"Z{zone_id}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
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


def process_video(video_path, grid_size=3):
    global processing_state
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator()
    
    frame_count = 0
    frame_times = []
    
    with processing_lock:
        processing_state['running'] = True
        processing_state['total_frames'] = total
        processing_state['frame_count'] = 0
    
    while processing_state['running']:
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
        
        pct = (frame_count / total) * 100 if total > 0 else 0
        cv2.putText(out_frame, f"[{frame_count}/{total}] {pct:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(out_frame, f"People: {len(people)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', out_frame)
        
        frame_times.append(time.time() - t0)
        if len(frame_times) > 30:
            frame_times.pop(0)
        
        with processing_lock:
            processing_state['frame'] = buffer.tobytes()
            processing_state['frame_count'] = frame_count
            processing_state['people_count'] = len(people)
            processing_state['zone_counts'] = zones
            processing_state['progress'] = pct
            if frame_times:
                processing_state['fps'] = len(frame_times) / sum(frame_times)
        
        frame_count += 1
    
    cap.release()
    with processing_lock:
        processing_state['running'] = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    grid_size = int(request.form.get('grid_size', 3))
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    thread = threading.Thread(target=process_video, args=(filepath, grid_size))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'processing'})


def video_stream():
    while True:
        with processing_lock:
            if processing_state['frame'] is None:
                time.sleep(0.1)
                continue
            frame_data = processing_state['frame']
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    with processing_lock:
        return jsonify({
            'running': processing_state['running'],
            'frame_count': processing_state['frame_count'],
            'total_frames': processing_state['total_frames'],
            'fps': round(processing_state['fps'], 1),
            'people_count': processing_state['people_count'],
            'zone_counts': processing_state['zone_counts'],
            'progress': round(processing_state['progress'], 1)
        })


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=7000)
