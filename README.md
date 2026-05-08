# Target-Aware Black Shirt Video Agent

CPU-first FastAPI project for local video analysis.

The system receives a video, detects people, estimates pose on each detected person crop, extracts the shirt region from shoulder and hip landmarks, checks whether the shirt region is black using OpenCV HSV rules, and marks matching people as `TARGET` in an annotated output video.

This project does not perform face recognition and does not identify a specific person. It detects a text-described visual attribute: `black shirt`.

## Pipeline

```text
Input video
  -> YOLO person detection
  -> crop each detected person
  -> MediaPipe pose estimation
  -> shirt polygon from shoulders and hips
  -> OpenCV black-color check inside shirt region
  -> draw target bbox + pose skeleton + shirt region
  -> save analyzed_video.mp4 + frame_report.json + frame_features.csv
```

## Why this is CPU-friendly

The project is designed for an older CPU-only laptop.

By default it does not run YOLO and pose on every single frame. It runs the heavy steps every few frames and reuses the last result between them. The output still contains overlays on every frame.

Default settings:

```text
resize_width = 640
process_every_n_frames = 5
max_seconds = 20
max_people = 3
yolo_model = yolov8n.pt
```

For better quality, use `process_every_n_frames=1` or `2`.
For faster processing, use `process_every_n_frames=5` or `10`.

## Local setup

Recommended Python version: Python 3.10.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or on Windows:

```bash
run_local.cmd
```

Open:

```text
http://localhost:8000/docs
```

Use the `/analyze-video` endpoint.

## Example request parameters

```text
target_query: black shirt
resize_width: 640
max_seconds: 20
process_every_n_frames: 5
min_match_score: 0.40
max_people: 3
use_yolo: true
draw_non_targets: false
```

## Output files

Each run creates a folder under `outputs/<run_id>/`:

```text
input_video.mp4
analyzed_video.mp4
frame_report.json
frame_features.csv
config.json
```

The API response includes direct local URLs such as:

```text
/outputs/<run_id>/analyzed_video.mp4
/outputs/<run_id>/frame_report.json
/outputs/<run_id>/frame_features.csv
```

## What is missing before first run

1. A real input video with one or more visible people.
2. At least one person should wear a clear black shirt.
3. Internet access may be required on first run so Ultralytics can download `yolov8n.pt`.
4. If the machine is offline, manually place `yolov8n.pt` in the `models/` folder.
5. This is not a trained custom model yet. It is a rule-based attribute detector built on top of pretrained YOLO and MediaPipe.

## Recommended demo video

Record a 5-15 second video with:

```text
one person wearing a black shirt
simple background
good lighting
person visible from head to hips
not too many dark objects in the shirt area
```

Then upload it in Swagger UI.

## Fallback mode without YOLO

If YOLO is slow or fails to install, set:

```text
use_yolo: false
```

The system will run MediaPipe pose on the full frame. This works best when there is only one main person in the video.

## Next step

After this local pipeline works, you can collect `frame_features.csv` files and train a small classifier in Colab to replace the rule-based color threshold with a learned target-matching or risk model.
