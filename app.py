"""Stable Streamlit entry point for the landmark review prototype."""

from pathlib import Path
import runpy
import sys

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

runpy.run_path(
    str(APP_DIR / "mediapipe_face_landmark_improved.py"),
    run_name="__main__",
)
