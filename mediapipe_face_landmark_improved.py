import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import json
import os
from pathlib import Path
from calibration import average_offsets, update_training

st.title("Improved Facial Landmark Tool")
st.caption("Review image-space landmarks and save coordinate calibration feedback. Detector weights are not retrained.")

ANALYSIS_MODE = st.selectbox("Choose Analysis Mode", ["Frontal Face", "Side Face"], index=0)

uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])

mp_face_mesh = mp.solutions.face_mesh

TRAIN_FILE = os.environ.get(
    "ORTHOMESH_FEEDBACK_PATH", str(Path(__file__).with_name("landmark_offsets.json"))
)


def load_training() -> dict:
    """Load accumulated offsets from disk."""
    if os.path.exists(TRAIN_FILE):
        with open(TRAIN_FILE, "r") as f:
            return json.load(f)
    return {}


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    h, w, _ = image_np.shape
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(image_np)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        if ANALYSIS_MODE == "Frontal Face":
            idx_map = {
                "Nasion": 168,
                "Subnasale": 94,
                "Menton": 152,
                "Glabella": 8
            }
        else:
            # Simple indices for profile example
            idx_map = {
                "Nasion": 168,
                "Pronasale": 1,
                "Pogonion": 152,
                "Orbitale": 33
            }
        training_data = load_training()
        offsets = average_offsets(training_data)
        raw_points = {name: (landmarks[i].x * w, landmarks[i].y * h) for name, i in idx_map.items()}
        auto_points = {}
        for name, (x, y) in raw_points.items():
            off_x, off_y = offsets.get(name, (0.0, 0.0))
            auto_points[name] = (x + off_x, y + off_y)

        st.subheader("Rate automatic detection")
        rating = st.radio("Are the automatic points correct?", ["Good", "Bad"], horizontal=True)
        st.subheader("Adjust Points Manually")
        adjusted_points = {}
        for name, (x, y) in auto_points.items():
            col1, col2 = st.columns(2)
            with col1:
                new_x = st.number_input(f"{name} X", value=float(x), key=f"{name}_x")
            with col2:
                new_y = st.number_input(f"{name} Y", value=float(y), key=f"{name}_y")
            adjusted_points[name] = (new_x, new_y)

        output = image_np.copy()
        for name, (x, y) in adjusted_points.items():
            cv2.circle(output, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.putText(output, name, (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        st.image(output, caption="Adjusted Landmarks", use_column_width=True)

        data = {"mode": ANALYSIS_MODE, "rating": rating, "points": adjusted_points}
        json_str = json.dumps(data, indent=2)
        if st.button("Save calibration feedback"):
            update_training(training_data, raw_points, adjusted_points)
            Path(TRAIN_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(TRAIN_FILE, "w") as f:
                json.dump(training_data, f, indent=2)
            st.success("Calibration offsets saved. MediaPipe weights are unchanged.")
        st.download_button("Download Points JSON", json_str, "landmarks.json", "application/json")
    else:
        st.error("Face not detected. Please try another image.")
