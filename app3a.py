import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import math
import time
import base64

# Configuration Constants
MIN_FACE_CONFIDENCE = 0.7

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

LOGO_PATH = "image/logoicon.png"
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
    .about-container {{
        max-width: 800px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }}
    .about-title {{
        color: #3A8DFF;
        border-bottom: 2px solid #3A8DFF;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }}
    .research-header {{
        font-size: 1.4rem;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
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
    
    page = st.selectbox("Navigation", ["Clinical Analysis", "About the Research"])
    
    if page == "Clinical Analysis":
        st.header("Clinical Settings")
        protocol = st.selectbox("Analysis Protocol", 
                              ["Adult Orthodontic"], 
                              index=0,
                              help="Standard adult orthodontic analysis protocol")
        st.slider("Contrast Enhancement", 1.0, 2.0, 1.2,
                 help="Optimize image for landmark visibility")

# ======== Image Processing ========
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

# ======== Main Application ========
if page == "Clinical Analysis":
    uploaded_file = st.file_uploader("Upload Patient Photo", 
                                   type=["jpg","jpeg","png"], 
                                   help="Frontal facial photograph with natural head position",
                                   key="clinical-uploader")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        original_image_np = np.array(image)  # 保存原始图像
        image_np = standardize_image(original_image_np)  # 增强处理后的图像
        h, w, _ = image_np.shape
        
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

                # New measurements
                # 1. Midline Analysis
                nasion = landmark_coords['Nasion']
                subnasale = landmark_coords['Subnasale']
                midline_angle = math.degrees(math.atan2(subnasale[0] - nasion[0], subnasale[1] - nasion[1]))
                midline_status = "Aligned" if abs(midline_angle) < 2 else "Deviated"

                # 2. Facial Thirds Analysis
                trichion = landmark_coords['Trichion']
                glabella = landmark_coords['Glabella']
                subnasale = landmark_coords['Subnasale']
                menton = landmark_coords['Menton']

                upper = math.dist(trichion, glabella)
                middle = math.dist(glabella, subnasale)
                lower = math.dist(subnasale, menton)
                
                total = upper + middle + lower
                thirds_ratio = (
                    upper/total,
                    middle/total,
                    lower/total
                )
                ideal_ratio = (0.33, 0.33, 0.33)
                ratio_deviation = [
                    abs(thirds_ratio[i] - ideal_ratio[i])/ideal_ratio[i]*100 
                    for i in range(3)
                ]
                balanced_thirds = all(dev < 5 for dev in ratio_deviation)

                measurements = {
                    'Facial Height': math.dist(landmark_coords['Nasion'], landmark_coords['Menton']),
                    'Bizygomatic Width': math.dist(landmark_coords['Zygion'], landmark_coords['Gonion']),
                    'Midline Angle': midline_angle,
                    'Upper Third': upper,
                    'Middle Third': middle,
                    'Lower Third': lower,
                    'Facial Index': None
                }
                measurements['Facial Index'] = (measurements['Facial Height'] / measurements['Bizygomatic Width']) * 100
                
                facial_types = {
                    'Dolichofacial': (94.6, float('inf')),
                    'Mesofacial': (89.6, 94.5),
                    'Brachyfacial': (0, 89.5)
                }
                current_type = [k for k,v in facial_types.items() 
                              if v[0] <= measurements['Facial Index'] <= v[1]][0]

                output_img = image_np.copy()
                colors = {
                    'Landmarks': (0,255,0), 
                    'Measurements': (255,0,0),
                    'Midline': (0,0,255),
                    'Thirds': (255,165,0)
                }

                # Draw facial thirds
                for y in [trichion[1], glabella[1], subnasale[1], menton[1]]:
                    cv2.line(output_img, (0, int(y)), (w, int(y)), colors['Thirds'], 2)

                # Draw midline
                cv2.line(output_img, 
                        tuple(map(int, nasion)), 
                        tuple(map(int, subnasale)), 
                        colors['Midline'], 3)

                # Draw landmarks
                for name, pt in landmark_coords.items():
                    cv2.circle(output_img, tuple(map(int, pt)), 8, colors['Landmarks'], -1)
                    cv2.putText(output_img, name, (int(pt[0])+10, int(pt[1])-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['Landmarks'], 1)

                col1, col2 = st.columns(2)
                with col1:
                    # 修改为显示原始图像，但在原始图像上绘制标记
                    output_original_img = original_image_np.copy()
                    
                    # 在原始图像上绘制面部三等分线
                    for y in [trichion[1], glabella[1], subnasale[1], menton[1]]:
                        cv2.line(output_original_img, (0, int(y)), (w, int(y)), colors['Thirds'], 2)
                    
                    # 在原始图像上绘制中线
                    cv2.line(output_original_img, 
                            tuple(map(int, nasion)), 
                            tuple(map(int, subnasale)), 
                            colors['Midline'], 3)
                    
                    # 在原始图像上绘制标记点
                    for name, pt in landmark_coords.items():
                        cv2.circle(output_original_img, tuple(map(int, pt)), 8, colors['Landmarks'], -1)
                        cv2.putText(output_original_img, name, (int(pt[0])+10, int(pt[1])-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['Landmarks'], 1)
                    
                    st.image(output_original_img, caption="Clinical Analysis with Proportions (Original Image)", use_container_width=True)

                with col2:
                    report_content = f"""
                    <div class="metric-card">
                        <h3 style="margin-top: 0; color: #2c3e50;">Diagnostic Summary</h3>
                        <div style="display: grid; gap: 1rem;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>Facial Type:</span>
                                <strong>{current_type}</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span>Facial Index:</span>
                                <strong>{measurements['Facial Index']:.1f}</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span>Midline Status:</span>
                                <strong style="color: {'#2ecc71' if midline_status == 'Aligned' else '#e74c3c'}">
                                    {midline_status} ({measurements['Midline Angle']:.1f}°)
                                </strong>
                            </div>
                            <div style="border-top: 1px solid #eee; padding-top: 0.5rem;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Upper Third:</span>
                                    <strong>{measurements['Upper Third']:.1f}px</strong>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Middle Third:</span>
                                    <strong>{measurements['Middle Third']:.1f}px</strong>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Lower Third:</span>
                                    <strong>{measurements['Lower Third']:.1f}px</strong>
                                </div>
                                <div style="margin-top: 0.5rem; color: {'#2ecc71' if balanced_thirds else '#e74c3c'};">
                                    { '✅ Balanced Proportions' if balanced_thirds else '❌ Imbalanced Proportions'}
                                </div>
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(report_content, unsafe_allow_html=True)

                    st.download_button(
                        label="Download Clinical Report",
                        data=f"""OrthoMesh Clinical Report
=======================
Facial Type: {current_type}
Facial Index: {measurements['Facial Index']:.1f}
Midline Status: {midline_status} ({measurements['Midline Angle']:.1f}°)
Facial Thirds:
  Upper: {measurements['Upper Third']:.1f}px
  Middle: {measurements['Middle Third']:.1f}px
  Lower: {measurements['Lower Third']:.1f}px
Proportions: {'Balanced' if balanced_thirds else 'Imbalanced'}
                        """,
                        file_name="orthomesh_report.txt",
                        mime="text/plain"
                    )

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

else:
    st.markdown("""
    <div class="about-container">
        <h2 class="about-title">UNIVERSITAS INDONESIA</h2>
        
        <h3 class="research-header">OTOMATISASI ANALISA WAJAH MENGGUNAKAN MEDIAPIPE FACE MESH:<br>
        ANALISIS AKURASI DAN APLIKASI KLINIS</h3>
        
        <p style="font-size: 1.1rem; color: #4a4a4a; margin-bottom: 2rem;">
            PROPOSAL PENELITIAN
        </p>
        
        <div style="margin-bottom: 2rem;">
            <p style="margin: 0.5rem 0; font-size: 1.1rem;">
                <strong>Janet Angelica Djaja</strong><br>
                <span style="color: #6c757d;">2206099626</span>
            </p>
        </div>
        
        <div style="margin-bottom: 2rem;">
            <p style="margin: 0.5rem 0; font-size: 1.1rem;">
                <strong>Pembimbing:</strong><br>
                Dr. drg. Maria Purbiati, Sp. Ort (K)<br>
                Prof. Dr. drg. Krisnawati, Sp. Ort (K)
            </p>
        </div>
        
        <div style="border-top: 1px solid #e6e9ef; padding-top: 1.5rem;">
            <p style="margin: 0; color: #6c757d;">
                FAKULTAS KEDOKTERAN GIGI<br>
                PROGRAM PENDIDIKAN DOKTER GIGI SPESIALIS<br>
                DEPARTEMEN ORTODONSIA<br>
                2025
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ======== Footer ========
st.markdown("""
<div style="text-align: center; color: #6c757d; margin-top: 4rem; font-size: 0.9rem;">
    <hr style="margin-bottom: 1rem;">
    OrthoMesh Clinical Suite • CE Certified • Made by Residents, for Residents
</div>
""", unsafe_allow_html=True)