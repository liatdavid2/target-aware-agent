import argparse
from pathlib import Path

from app.analyzer import VideoAnalyzer
from app.config import AnalyzerConfig


def main():
    parser = argparse.ArgumentParser(description="Analyze a video and mark people wearing a black shirt.")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--target-query", default="black shirt")
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--process-every-n-frames", type=int, default=5)
    parser.add_argument("--min-match-score", type=float, default=0.40)
    parser.add_argument("--max-people", type=int, default=3)
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--draw-non-targets", action="store_true")
    args = parser.parse_args()

    config = AnalyzerConfig(
        target_query=args.target_query,
        resize_width=args.resize_width,
        max_seconds=args.max_seconds,
        process_every_n_frames=args.process_every_n_frames,
        min_match_score=args.min_match_score,
        max_people=args.max_people,
        use_yolo=not args.no_yolo,
        draw_non_targets=args.draw_non_targets,
    )

    analyzer = VideoAnalyzer(config)
    summary = analyzer.analyze(Path(args.input))
    print(summary)


if __name__ == "__main__":
    main()
