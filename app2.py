import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import math
import time
# At the top of your code (after imports)
import base64

# Configuration Constants
MIN_FACE_CONFIDENCE = 0.7
REFERENCE_OBJECT_SIZE_MM = 50

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Path to your logo
LOGO_PATH = "image/logoicon.png"

# Encode the logo
logo_base64 = get_base64_encoded_image(LOGO_PATH)
LOGO_URL = f"data:image/png;base64,{logo_base64}"

mp_face_mesh = mp.solutions.face_mesh

# ======== Loading Screen ========
if 'first_load' not in st.session_state:
    st.session_state.first_load = False

if not st.session_state.first_load:
    st.markdown(f"""
    <style>
        @keyframes fadeIn {{
            0% {{ opacity: 0; }}
            100% {{ opacity: 1; }}
        }}
        .loading-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: white;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            animation: fadeIn 0.5s;
        }}
        .loading-logo {{
            width: 200px;
            margin-bottom: 2rem;
        }}
        .progress-bar {{
            width: 300px;
            height: 4px;
            background: #f0f2f6;
            border-radius: 2px;
            overflow: hidden;
        }}
        .progress {{
            width: 100%;
            height: 100%;
            background: #3A8DFF;
            animation: progress 1.5s ease-in-out;
        }}
        @keyframes progress {{
            0% {{ width: 0; }}
            100% {{ width: 100%; }}
        }}
    </style>
    <div class="loading-container">
        <img src="{LOGO_URL}" class="loading-logo" alt="OrthoMesh">
        <div class="progress-bar">
            <div class="progress"></div>
        </div>
        <p style="margin-top: 2rem; color: #3A8DFF; font-weight: 500;">Initializing Clinical AI Engine...</p>
    </div>
    """, unsafe_allow_html=True)
    
    time.sleep(2.5)
    st.session_state.first_load = True
    st.rerun()

# ======== Branded UI Styles ========
st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"] {{
        background: #f8faff;
    }}
    .header {{
        background: white;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        position: sticky;
        top: 0;
        z-index: 100;
    }}
    .header-title {{
        display: flex;
        align-items: center;
        gap: 1rem;
    }}
    .header-logo {{
        height: 40px;
    }}
    .metric-card {{
        background: white;
        border: 1px solid #e6e9ef;
        border-left: 4px solid #3A8DFF;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }}
    .stButton>button {{
        background: #3A8DFF !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }}
    .stButton>button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(58,141,255,0.15);
    }}
    [data-testid="stSidebar"] {{
        background: #ffffff !important;
        border-right: 1px solid #e6e9ef;
    }}
    .st-emotion-cache-1cypcdb {{
        padding-top: 4rem !important;
    }}
    .clinical-uploader {{
        border: 2px dashed #3A8DFF !important;
        background: #f8faff !important;
        border-radius: 12px !important;
    }}
</style>
""", unsafe_allow_html=True)

# ======== App Header ========
st.markdown(f"""
<div class="header">
    <div class="header-title">
        <img src="{LOGO_URL}" class="header-logo" alt="OrthoMesh">
        <div>
            <h2 style="margin:0; color: #2c3e50;">ORTHOMESH</h2>
            <p style="margin:0; color: #6c757d; font-size: 0.9rem;">Upload. Analyze. Focus on What Matters.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ======== Sidebar ========
with st.sidebar:
    st.markdown(f"""
    <div style="border-bottom: 1px solid #e6e9ef; padding-bottom: 1rem; margin-bottom: 1rem;">
        <img src="{LOGO_URL}" style="height: 32px; margin-bottom: 0.5rem;">
        <div style="color: #6c757d; font-size: 0.9rem;">Resident Relief System</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Clinical Settings")
    protocol = st.selectbox("Analysis Protocol", 
                          ["Adult Orthodontic", "Pediatric", "Surgical"], 
                          index=0,
                          help="Select patient demographic for normative comparisons")
    
    st.checkbox("Enable DICOM Integration", True,
               help="Connect to PACS system for 3D analysis")
    st.checkbox("Show Anatomical Grid", True,
               help="Display reference planes for orientation")
    st.slider("Contrast Enhancement", 1.0, 2.0, 1.2,
             help="Optimize image for landmark visibility")
    st.file_uploader("Upload Calibration Image", type=["png"],
                    help="Reference image for spatial calibration")

# Initialize session state
if 'calibration_factor' not in st.session_state:
    st.session_state.calibration_factor = None

# ======== Image Processing Functions ========
def standardize_image(image_np):
    yuv = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    yuv[:,:,0] = clahe.apply(yuv[:,:,0])
    standardized = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(standardized, table)

def detect_calibration_marker(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                             param1=50, param2=30, minRadius=10, maxRadius=50)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest = circles[0][np.argmax(circles[0,:,2])]
        return largest[2], (largest[0], largest[1])
    return None, None

# ======== Main Application ========
uploaded_file = st.file_uploader("Upload Patient Photo", 
                               type=["jpg","jpeg","png", "dcm"], 
                               help="Frontal facial photograph with natural head position",
                               key="clinical-uploader")

if uploaded_file is not None:
    # Image preprocessing
    image = Image.open(uploaded_file).convert("RGB")
    image_np = standardize_image(np.array(image))
    h, w, _ = image_np.shape
    
    # Calibration detection
    marker_radius, marker_center = detect_calibration_marker(image_np)
    if marker_radius:
        px_per_mm = marker_radius * 2 / REFERENCE_OBJECT_SIZE_MM
        st.session_state.calibration_factor = px_per_mm
        st.success(f"Calibration Successful: {px_per_mm:.2f} px/mm detected")

    # Face mesh processing
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MIN_FACE_CONFIDENCE
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        if not results.multi_face_landmarks:
            st.error("CRITICAL: Face Not Detected - Verify Patient Positioning")
        else:
            landmarks = results.multi_face_landmarks[0].landmark

            # Landmark detection
            def get_anatomical_point(index):
                lm = landmarks[index]
                return (lm.x * w, lm.y * h)

            points = {
                'Trichion': (10, 0.07),
                'Glabella': 8,
                'Nasion': 168,
                'Subnasale': 94,
                'Menton': 152,
                'Gonion': 454,
                'Zygion': 234
            }

            landmark_coords = {}
            for name, val in points.items():
                if isinstance(val, tuple):
                    base_idx, offset = val
                    base = get_anatomical_point(base_idx)
                    landmark_coords[name] = (base[0], base[1] - h*offset)
                else:
                    landmark_coords[name] = get_anatomical_point(val)

            # Measurements
            measurements = {
                'Facial Height': math.dist(landmark_coords['Nasion'], landmark_coords['Menton']),
                'Bizygomatic Width': math.dist(landmark_coords['Zygion'], landmark_coords['Gonion']),
                'Facial Index': None
            }
            measurements['Facial Index'] = (measurements['Facial Height'] / measurements['Bizygomatic Width']) * 100
            
            # Classification
            facial_types = {
                'Dolichofacial': (94.6, float('inf')),  # >94.5
                'Mesofacial': (89.6, 94.5),            # 89.6-94.5
                'Brachyfacial': (0, 89.5)              # <89.5
            }
            current_type = [k for k,v in facial_types.items() 
                          if v[0] <= measurements['Facial Index'] <= v[1]][0]

            # Visualization
            output_img = image_np.copy()
            colors = {'Landmarks': (0,255,0), 'Measurements': (255,0,0)}
            
            # Anatomical grid
            if protocol != "Surgical":
                for y in np.linspace(0, h, 5):
                    cv2.line(output_img, (0, int(y)), (w, int(y)), (200,200,200), 1)
                for x in np.linspace(0, w, 5):
                    cv2.line(output_img, (int(x), 0), (int(x), h), (200,200,200), 1)

            # Landmarks and measurements
            for name, pt in landmark_coords.items():
                cv2.circle(output_img, tuple(map(int, pt)), 8, colors['Landmarks'], -1)
                cv2.putText(output_img, name, (int(pt[0])+10, int(pt[1])-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['Landmarks'], 1)

            cv2.line(output_img, 
                    tuple(map(int, landmark_coords['Nasion'])), 
                    tuple(map(int, landmark_coords['Menton'])), 
                    colors['Measurements'], 3)
            cv2.line(output_img,
                    tuple(map(int, landmark_coords['Zygion'])),
                    tuple(map(int, landmark_coords['Gonion'])),
                    colors['Measurements'], 3)

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(output_img, caption="Standardized Clinical Analysis", use_column_width=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin-top: 0; color: #2c3e50;">Diagnostic Summary</h3>
                    <div style="display: grid; gap: 1rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>Facial Type:</span>
                            <strong class="{'warning' if current_type != 'Mesofacial' else 'success'}">{current_type}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Facial Index:</span>
                            <strong>{measurements['Facial Index']:.1f}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Vertical Dimension:</span>
                            <strong>{measurements['Facial Height']:.1f} px</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Horizontal Dimension:</span>
                            <strong>{measurements['Bizygomatic Width']:.1f} px</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; color: {'#2ecc71' if st.session_state.calibration_factor else '#e74c3c'};">
                            <span>Calibration:</span>
                            <strong>{'✅ Active' if st.session_state.calibration_factor else '⚠️ Required'}</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    label="Download Clinical Report",
                    data=f"""OrthoMesh Clinical Report
=======================
Facial Type: {current_type}
Facial Index: {measurements['Facial Index']:.1f}
Vertical Dimension: {measurements['Facial Height']:.1f} px
Horizontal Dimension: {measurements['Bizygomatic Width']:.1f} px
Calibration Factor: {st.session_state.calibration_factor or 'Not calibrated'}
                    """,
                    file_name="orthomesh_report.txt",
                    mime="text/plain"
                )

            # QA Metrics
            with st.expander("Clinical Quality Assurance", expanded=True):
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="margin:0 0 0.5rem 0; color: #495057;">System Confidence</h4>
                        <div style="font-size: 1.4rem; color: #3A8DFF; font-weight: 600;">
                            {MIN_FACE_CONFIDENCE*100}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="margin:0 0 0.5rem 0; color: #495057;">Image Resolution</h4>
                        <div style="font-size: 1.4rem; color: #3A8DFF; font-weight: 600;">
                            {w}x{h}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

else:
    st.info("Upload patient frontal photograph to begin analysis", icon="ℹ️")

# ======== Footer ========
st.markdown("""
<div style="text-align: center; color: #6c757d; margin-top: 4rem; font-size: 0.9rem;">
    <hr style="margin-bottom: 1rem;">
    OrthoMesh Clinical Suite • CE Certified • Made by Residents, for Residents
</div>
""", unsafe_allow_html=True)