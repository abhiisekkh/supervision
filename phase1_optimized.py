#!/usr/bin/env python3
"""
OPTIMIZED Phase 1 - Improved ByteTrack ID Stability
MacBook M1 optimized with stable ID parameters
"""

import subprocess
import sys
from pathlib import Path

VIDEO_PATH = "/Users/abhisekh/Desktop/Major Project/Video/Dataset/Videos/6387-191695740.mp4"
WORKSPACE = Path("/Users/abhisekh/Desktop/supervision")

print("""
╔════════════════════════════════════════════════════════════════╗
║         PHASE 1 OPTIMIZED: STABLE ID TRACKING                 ║
║     Reprocessing: 6387-191695740.mp4 with M1 optimization      ║
╚════════════════════════════════════════════════════════════════╝

🔧 OPTIMIZATIONS FOR STABLE IDs:
  • track_activation_threshold: 0.20 → 0.40 (stricter activation)
  • lost_track_buffer: 40 → 150 frames (keep IDs alive longer)
  • minimum_matching_threshold: 0.85 → 0.80 (forgiving matching)
  • conf: 0.20 → 0.35 (higher detection quality)
  • imgsz: 960 → 1280 (better small person detection)
  • M1 optimization: Using tiled inference for dense crowds

💡 Why These Changes Fix ID Issues:
  1. Stricter activation prevents premature ID creation
  2. Longer buffer keeps IDs during occlusions/gaps
  3. Forgiving matching connects detections to existing tracks
  4. Higher confidence reduces false positives
  5. Better image size catches small people
""")

# Optimized command for Phase 1
cmd = [
    "python3",
    "detect_people.py",
    VIDEO_PATH,
    "--no-preview",
    # Model & Detection
    "--model", "yolov8m.pt",
    "--conf", "0.35",           # ← INCREASED from 0.20
    "--imgsz", "1280",          # ← INCREASED from 960  
    "--max-det", "2000",        # ← Allow more detections for dense crowds
    # ByteTrack - STABLE ID PARAMETERS
    "--track-activation-threshold", "0.40",  # ← INCREASED from 0.20
    "--lost-track-buffer", "150",             # ← INCREASED from 40
    "--minimum-matching-threshold", "0.80",  # ← DECREASED from 0.85
    # Optical Flow
    "--flow-interval", "3",     # ← Compute more often
    # Smoothing
    "--smoothing-alpha", "0.50", # ← More stable smoothing
    # M1 Optimization
    "--use-slicer",             # ← Tiled inference
    "--slice-size", "960",      # ← Optimal for M1
    "--slice-overlap", "100",
    # Stable ID parameters (if available)
    "--stable-id-gap-seconds", "1.5",
    "--stable-id-distance-ratio", "0.08",
    "--stable-id-min-iou", "0.10",
]

print(f"⏳ Starting optimized Phase 1 processing...\n")
print(f"Command: python3 detect_people.py [video] \\")
for i in range(1, len(cmd), 2):
    if i < len(cmd) - 1:
        print(f"  {cmd[i]} {cmd[i+1]} \\")
    else:
        print(f"  {cmd[i]}")

try:
    result = subprocess.run(
        cmd,
        cwd=str(WORKSPACE),
        timeout=None
    )
    
    if result.returncode == 0:
        print("""
╔════════════════════════════════════════════════════════════════╗
║                  ✅ PHASE 1 COMPLETED!                        ║
║         Video with stable IDs is ready for review             ║
╚════════════════════════════════════════════════════════════════╝
        """)
    else:
        print(f"\n❌ Phase 1 failed with return code {result.returncode}")
        sys.exit(1)
        
except KeyboardInterrupt:
    print("\n⏹️ Process interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
