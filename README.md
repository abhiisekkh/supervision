# Crowd Flow Analysis Tool

Real-time person detection, tracking, and movement analysis.

## Quick Start

```bash
uv sync
uv run python detect_people.py
```

Pick a video. Processing starts automatically.

## Output

```
video_name/
├── video_name_result.mp4    # Annotated video
└── flow_vectors.csv         # Tracking data
```

## Features

✓ Person detection (YOLOv8)
✓ ID tracking (ByteTrack)
✓ Movement arrows with speed coloring
✓ Grid zones with congestion status
✓ CSV export for ML models

## Controls

- **P** - Pause/Resume
- **Q** - Quit

## CSV Data

```
frame,tracker_id,cx,cy,arrow_dx,arrow_dy,speed,direction_degrees,zone_number,congestion_status
```

Ready for machine learning!
