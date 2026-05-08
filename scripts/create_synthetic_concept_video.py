"""
Create a synthetic concept video for README/demo explanations.

This video is not intended to test YOLO, because YOLO is trained on real images,
not simple drawings. Use a real phone/webcam video to test the actual pipeline.
"""

import cv2
import numpy as np
from pathlib import Path

out_dir = Path("synthetic_demo")
out_dir.mkdir(exist_ok=True)
output_path = out_dir / "black_shirt_concept_input.mp4"

width, height = 960, 540
fps = 24
frames = 120
writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

for i in range(frames):
    frame = np.full((height, width, 3), (45, 45, 50), dtype=np.uint8)
    x = 150 + int(500 * i / frames)
    # Person-like drawing with black shirt
    cv2.circle(frame, (x, 180), 28, (210, 190, 170), -1)
    cv2.rectangle(frame, (x - 45, 220), (x + 45, 330), (20, 20, 20), -1)
    cv2.rectangle(frame, (x - 40, 330), (x - 8, 460), (60, 60, 90), -1)
    cv2.rectangle(frame, (x + 8, 330), (x + 40, 460), (60, 60, 90), -1)
    cv2.putText(frame, "Synthetic concept only - use real video for YOLO", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
    writer.write(frame)

writer.release()
print(output_path)
