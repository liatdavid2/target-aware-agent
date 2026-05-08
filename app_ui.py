import os
import time
import subprocess
from pathlib import Path

import gradio as gr
import requests


API_URL = "http://127.0.0.1:8000/analyze-video"
OUTPUTS_DIR = Path("outputs")


def find_latest_output_video() -> str | None:
    if not OUTPUTS_DIR.exists():
        return None

    videos = list(OUTPUTS_DIR.rglob("analyzed_video.mp4"))

    if not videos:
        return None

    latest = max(videos, key=lambda p: p.stat().st_mtime)
    return str(latest)


def analyze_video(
    video_path,
    target_query,
    resize_width,
    max_seconds,
    process_every_n_frames,
    min_match_score,
    max_people,
    use_yolo,
    draw_non_targets,
    enable_avatar,
):
    if video_path is None:
        return None, "Please upload a video."

    with open(video_path, "rb") as f:
        files = {
            "file": (os.path.basename(video_path), f, "video/mp4")
        }

        data = {
            "target_query": target_query,
            "resize_width": int(resize_width),
            "max_seconds": float(max_seconds),
            "process_every_n_frames": int(process_every_n_frames),
            "min_match_score": float(min_match_score),
            "max_people": int(max_people),
            "use_yolo": bool(use_yolo),
            "draw_non_targets": bool(draw_non_targets),
            "enable_avatar": bool(enable_avatar),
        }

        response = requests.post(API_URL, files=files, data=data, timeout=600)

    if response.status_code != 200:
        return None, f"API error: {response.status_code}\n{response.text}"

    time.sleep(0.5)

    output_video = find_latest_output_video()

    if output_video is None:
        return None, "Analysis completed, but analyzed_video.mp4 was not found."

    return output_video, "Done."


def build_ui():
    with gr.Blocks(title="Target-Aware Video Agent") as demo:
        gr.Markdown("# Target-Aware Video Agent")
        gr.Markdown("Upload a video, choose a target such as `black shirt`, and generate an analyzed output video.")

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Input video")

                target_query = gr.Textbox(
                    label="Target query",
                    value="black shirt",
                    placeholder="black shirt / yellow shirt / חולצה שחורה",
                )

                enable_avatar = gr.Checkbox(
                    label="Enable pose avatar",
                    value=True,
                )

                use_yolo = gr.Checkbox(
                    label="Use YOLO person detection",
                    value=True,
                )

                draw_non_targets = gr.Checkbox(
                    label="Draw non-target people",
                    value=False,
                )

                resize_width = gr.Slider(
                    label="Resize width",
                    minimum=320,
                    maximum=960,
                    step=80,
                    value=640,
                )

                max_seconds = gr.Slider(
                    label="Max seconds",
                    minimum=3,
                    maximum=60,
                    step=1,
                    value=20,
                )

                process_every_n_frames = gr.Slider(
                    label="Process every N frames",
                    minimum=1,
                    maximum=15,
                    step=1,
                    value=5,
                )

                min_match_score = gr.Slider(
                    label="Min match score",
                    minimum=0.05,
                    maximum=0.90,
                    step=0.05,
                    value=0.40,
                )

                max_people = gr.Slider(
                    label="Max people",
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=3,
                )

                run_button = gr.Button("Analyze video")

            with gr.Column():
                output_video = gr.Video(label="Analyzed output video")
                status = gr.Textbox(label="Status")

        run_button.click(
            fn=analyze_video,
            inputs=[
                video_input,
                target_query,
                resize_width,
                max_seconds,
                process_every_n_frames,
                min_match_score,
                max_people,
                use_yolo,
                draw_non_targets,
                enable_avatar,
            ],
            outputs=[output_video, status],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860)