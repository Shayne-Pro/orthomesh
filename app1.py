import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import math

mp_face_mesh = mp.solutions.face_mesh

# Custom CSS for professional styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .highlight {
        color: #2ecc71;
        font-weight: bold;
    }
    .warning {
        color: #e74c3c;
        font-weight: bold;
    }
    .section-title {
        color: #3498db;
        font-size: 1.4em;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("Orthodontic Facial Analysis System")
st.markdown("""
**Clinical Facial Analysis System**  
*Automated Cephalometric Landmark Identification and Facial Type Classification*
""")

# ======== Sidebar Controls ========
with st.sidebar:
    st.header("Configuration")
    offset_percent = st.slider("Trichion Offset (%)", 5, 10, 7)
    st.markdown("---")
    st.info("""
    **Landmark Guide**:
    - Trichion: Hairline point
    - Glabella: Foremost point between eyebrows
    - Subnasale: Base of nasal septum
    - Menton: Lowest chin point
    """)

# ======== Main Processing ========
uploaded_file = st.file_uploader("Upload Frontal Facial Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    h, w, _ = image_np.shape

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(img_bgr)

        if not results.multi_face_landmarks:
            st.error("Face not detected. Please ensure frontal positioning and good lighting.")
        else:
            landmarks = results.multi_face_landmarks[0].landmark

            # ======== Landmark Processing ========
            def get_pixel_coords(idx):
                lm = landmarks[idx]
                return (lm.x * w, lm.y * h)

            # Key landmarks
            trichion = get_pixel_coords(10)
            glabella = get_pixel_coords(8)
            subnasale = get_pixel_coords(94)
            menton = get_pixel_coords(152)
            nasion = get_pixel_coords(168)
            bizy_right = get_pixel_coords(234)
            bizy_left = get_pixel_coords(454)

            # Adjust trichion with offset
            trichion = (trichion[0], max(0, trichion[1] - h*offset_percent/100))

            # ======== Facial Measurements ========
            # Vertical dimensions
            facial_height = math.dist(nasion, menton)
            upper_face = math.dist(trichion, glabella)
            middle_face = math.dist(glabella, subnasale)
            lower_face = math.dist(subnasale, menton)
            
            # Horizontal dimensions
            bizy_width = math.dist(bizy_left, bizy_right)
            
            # Facial index calculation
            facial_index = (facial_height / bizy_width) * 100
            
            # Facial type classification
            if facial_index > 104.9:
                face_type = "Dolichofacial (Long Face)"
                face_color = "#e74c3c"
            elif 94.5 <= facial_index <= 104.9:
                face_type = "Mesofacial (Average)"
                face_color = "#2ecc71"
            else:
                face_type = "Brachyfacial (Short Face)"
                face_color = "#f1c40f"

            # ======== Visualization ========
            output_img = img_bgr.copy()
            
            # Draw landmarks and lines
            for pt, color in [(trichion, (0,255,0)), (glabella, (0,255,255)),
                            (subnasale, (255,0,0)), (menton, (255,0,255))]:
                cv2.circle(output_img, tuple(map(int, pt)), 8, color, -1)
            
            # Draw bizygomatic box
            cv2.rectangle(output_img, 
                         (int(min(bizy_left[0], bizy_right[0])), int(nasion[1])),
                         (int(max(bizy_left[0], bizy_right[0])), int(menton[1])),
                         (0,165,255), 2)

            # Draw facial axis
            cv2.line(output_img, tuple(map(int, nasion)), tuple(map(int, menton)), 
                    (255,255,0), 2)

            # Convert to RGB for Streamlit
            output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.image(output_rgb, caption="Analysis Results", use_column_width=True)

            # ======== Clinical Report ========
            st.markdown("---")
            st.markdown(f'<h2 class="section-title">Clinical Report</h2>', unsafe_allow_html=True)
            
            # Facial Type Card
            st.markdown(f"""
            <div class="metric-card">
                <h3>Facial Type Classification</h3>
                <p style='font-size: 1.5em; color: {face_color};'>{face_type}</p>
                <p>Facial Index: {facial_index:.1f} (Nasion-Menton/Bizygomatic × 100)</p>
                <small>Classification Reference:<br>
                Dolichofacial: >104.9 | Mesofacial: 94.5-104.9 | Brachyfacial: <94.5</small>
            </div>
            """, unsafe_allow_html=True)

            # Measurement Cards
            col3, col4, col5 = st.columns(3)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Vertical Proportions</h4>
                    <p>Upper Face: {upper_face:.1f}px</p>
                    <p>Middle Face: {middle_face:.1f}px</p>
                    <p>Lower Face: {lower_face:.1f}px</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Horizontal Measurements</h4>
                    <p>Bizygomatic Width: {bizy_width:.1f}px</p>
                    <p>Facial Height: {facial_height:.1f}px</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Proportional Analysis</h4>
                    <p>Upper/Middle Ratio: {(upper_face/middle_face):.2f}</p>
                    <p>Middle/Lower Ratio: {(middle_face/lower_face):.2f}</p>
                </div>
                """, unsafe_allow_html=True)

            # Clinical Interpretation
            st.markdown(f"""
            <div class="metric-card">
                <h4>Clinical Interpretation</h4>
                <p>Facial Type: <span style="color:{face_color};">{face_type.split(' ')[0]}</span></p>
                <p>Vertical Dimension: {'Increased' if facial_index > 104.9 else 'Reduced' if facial_index < 94.5 else 'Normal'}</p>
                <p>Horizontal Development: {'Narrow' if facial_index > 104.9 else 'Broad' if facial_index < 94.5 else 'Proportional'}</p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("Please upload a frontal facial photograph for analysis.")