"""Stable Streamlit entry point for the landmark review prototype."""

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).with_name("mediapipe_face_landmark_improved.py")),
    run_name="__main__",
)
