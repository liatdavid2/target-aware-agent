from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class AnalyzerConfig:
    target_query: str = "black shirt"
    resize_width: int = 640
    max_seconds: float = 20.0
    process_every_n_frames: int = 5
    min_match_score: float = 0.40
    max_people: int = 3
    use_yolo: bool = True
    draw_non_targets: bool = False
    yolo_confidence: float = 0.35
    yolo_model_name: str = "yolov8n.pt"
