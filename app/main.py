import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles

from app.analyzer import VideoAnalyzer
from app.config import AnalyzerConfig, OUTPUTS_DIR

app = FastAPI(title="Target-Aware Black Shirt Video Agent")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze-video")
def analyze_video(
    file: UploadFile = File(...),
    target_query: str = Form("black shirt"),
    resize_width: int = Form(640),
    max_seconds: float = Form(20.0),
    process_every_n_frames: int = Form(5),
    min_match_score: float = Form(0.40),
    max_people: int = Form(3),
    use_yolo: bool = Form(True),
    draw_non_targets: bool = Form(False),
    yolo_confidence: float = Form(0.35),
    enable_avatar: bool = Form(False),
    avatar_person_position: str = Form("right"),
    avatar_person_rank_from_right: int = Form(1),
):
    upload_dir = OUTPUTS_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "input.mp4").suffix or ".mp4"
    upload_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    config = AnalyzerConfig(
        target_query=target_query,
        resize_width=resize_width,
        max_seconds=max_seconds,
        process_every_n_frames=process_every_n_frames,
        min_match_score=min_match_score,
        max_people=max_people,
        use_yolo=use_yolo,
        draw_non_targets=draw_non_targets,
        yolo_confidence=yolo_confidence,
        enable_avatar=enable_avatar,
        avatar_person_position=avatar_person_position,
        avatar_person_rank_from_right=avatar_person_rank_from_right,
    )
    analyzer = VideoAnalyzer(config)
    summary = analyzer.analyze(upload_path)

    run_id = summary["run_id"]
    return {
        "status": "completed",
        "run_id": run_id,
        "frames_processed": summary["frames_processed"],
        "heavy_model_frames": summary["heavy_model_frames"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "effective_fps": summary["effective_fps"],
        "output_video_url": f"/outputs/{run_id}/analyzed_video.mp4",
        "frame_report_url": f"/outputs/{run_id}/frame_report.json",
        "frame_features_url": f"/outputs/{run_id}/frame_features.csv",
    }
