import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import math
import time
import base64
# Removed FPDF imports
import io # Still needed for logo processing if used elsewhere, but not for PDF
from datetime import datetime # Added for report date filename
# Removed tempfile import

# Configuration Constants
MIN_FACE_CONFIDENCE = 0.7 # Although not displayed, still used for detection

# --- Helper Functions ---

def get_base64_encoded_image(image_path):
    """Reads an image file and returns its base64 encoded string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        # Log warning instead of crashing the app if logo is missing
        print(f"Warning: Logo file not found at {image_path}")
        return None

def standardize_image(image_np):
    """Applies CLAHE and Gamma Correction for image standardization."""
    # Convert to YUV color space to apply CLAHE to the Luma channel (Y)
    yuv = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)
    # Create CLAHE object (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    # Apply CLAHE to the Y channel
    yuv[:,:,0] = clahe.apply(yuv[:,:,0])
    # Convert back to RGB
    standardized = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    # Apply Gamma Correction
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    # Build a lookup table mapping pixel values [0, 255] to gamma corrected values
    table = np.array([((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # Apply gamma correction using the lookup table
    return cv2.LUT(standardized, table)

# --- PDF Generation Class and Function REMOVED ---

# --- Streamlit App Setup ---

# Load Logo
LOGO_PATH = "image/logoicon.png"
logo_base64 = get_base64_encoded_image(LOGO_PATH)
LOGO_URL = f"data:image/png;base64,{logo_base64}" if logo_base64 else ""

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Initialize session state for patient info persistence
default_patient_info = {
    "name": "", "age": 0, "mrn": "", "gender": "Female", "info_complete": False
}
for key, default_value in default_patient_info.items():
    if f"patient_{key}" not in st.session_state:
        st.session_state[f"patient_{key}"] = default_value

# Loading Screen Logic
if 'first_load' not in st.session_state: st.session_state.first_load = False
if not st.session_state.first_load and LOGO_URL:
    st.markdown(f""" <style>@keyframes fadeIn {{ 0% {{ opacity: 0; }} 100% {{ opacity: 1; }} }} .loading-container {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: white; z-index: 9999; display: flex; flex-direction: column; align-items: center; justify-content: center; animation: fadeIn 0.5s; }} .loading-logo {{ width: 200px; margin-bottom: 2rem; }} .progress-bar {{ width: 300px; height: 4px; background: #f0f2f6; border-radius: 2px; overflow: hidden; }} .progress {{ width: 100%; height: 100%; background: #3A8DFF; animation: progress 1.5s ease-in-out; }} @keyframes progress {{ 0% {{ width: 0; }} 100% {{ width: 100%; }} }} </style> <div class="loading-container"><img src="{LOGO_URL}" class="loading-logo" alt="OrthoMesh"><div class="progress-bar"><div class="progress"></div></div><p style="margin-top: 2rem; color: #3A8DFF; font-weight: 500;">Initializing Clinical AI Engine...</p></div> """, unsafe_allow_html=True)
    time.sleep(2.5); st.session_state.first_load = True; st.rerun()
elif not st.session_state.first_load: st.session_state.first_load = True

# Branded UI Styles (CSS)
st.markdown(f""" <style> [data-testid="stAppViewContainer"] {{ background: #f8faff; }} .header {{ background: white; padding: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); position: sticky; top: 0; z-index: 100; }} .header-title {{ display: flex; align-items: center; gap: 1rem; }} .header-logo {{ height: 40px; {'display: none;' if not LOGO_URL else ''} }} .metric-card {{ background: white; border: 1px solid #e6e9ef; border-left: 4px solid #3A8DFF; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 6px rgba(0,0,0,0.03); }} .stButton>button {{ background: #3A8DFF !important; border-radius: 8px !important; font-weight: 500 !important; transition: all 0.2s !important; }} .stButton>button:hover {{ transform: translateY(-1px); box-shadow: 0 3px 8px rgba(58,141,255,0.15); }} [data-testid="stSidebar"] {{ background: #ffffff !important; border-right: 1px solid #e6e9ef; }} .st-emotion-cache-1cypcdb {{ padding-top: {'1rem' if not LOGO_URL else '4rem'} !important; }} .clinical-uploader label {{ font-size: 1.1rem !important; font-weight: 500 !important; }} .clinical-uploader > div > div {{ border: 2px dashed #3A8DFF !important; background: #f0f5ff !important; border-radius: 12px !important; }} .about-container {{ max-width: 800px; margin: 2rem auto; padding: 2rem; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }} .about-title {{ color: #3A8DFF; border-bottom: 2px solid #3A8DFF; padding-bottom: 0.5rem; margin-bottom: 1.5rem; }} .research-header {{ font-size: 1.4rem; color: #2c3e50; margin: 1.5rem 0 1rem 0; }} .stMarkdown pre {{ display: none; }} [data-testid="stTextInput"], [data-testid="stNumberInput"], [data-testid="stSelectbox"] {{ margin-bottom: 0.5rem; }} </style> """, unsafe_allow_html=True)

# App Header
st.markdown(f""" <div class="header"> <div class="header-title"> <img src="{LOGO_URL}" class="header-logo" alt="OrthoMesh"> <div> <h2 style="margin:0; color: #2c3e50;">ORTHOMESH</h2> <p style="margin:0; color: #6c757d; font-size: 0.9rem;">Upload. Analyze. Focus on What Matters.</p> </div> </div> </div> """, unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown(f""" <div style="border-bottom: 1px solid #e6e9ef; padding-bottom: 1rem; margin-bottom: 1rem;"> {f'<img src="{LOGO_URL}" style="height: 32px; margin-bottom: 0.5rem;">' if LOGO_URL else ''} <div style="color: #6c757d; font-size: 0.9rem;">Orthomesh</div> </div> """, unsafe_allow_html=True)
    page = st.selectbox("Navigation", ["Clinical Analysis", "About the Research"])
    if page == "Clinical Analysis":
        st.header("Clinical Settings"); protocol = st.selectbox("Analysis Protocol", ["Adult Orthodontic"], index=0, help="Standard adult orthodontic analysis protocol")
        # contrast_value = st.slider("Contrast Enhancement (Visual Only)", 1.0, 2.0, 1.2, help="Note: Fixed standardization is applied during processing. This slider is for potential future use.")

# --- Main Page Content ---

if page == "Clinical Analysis":

    # Step 1: Patient Information Input
    st.subheader("1. Enter Patient Information")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        # Bind inputs directly to session state keys
        st.session_state.patient_name = st.text_input("Patient Name", st.session_state.patient_name)
        st.session_state.patient_age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.patient_age if st.session_state.patient_age > 0 else 0, step=1)
    with col_info2:
        st.session_state.patient_mrn = st.text_input("Medical Record Number (MRN)", st.session_state.patient_mrn)
        st.session_state.patient_gender = st.selectbox("Gender", ["Female", "Male", "Other"], index=["Female", "Male", "Other"].index(st.session_state.patient_gender) if st.session_state.patient_gender in ["Female", "Male", "Other"] else 0)

    # Validate if all required info is provided
    info_provided = bool(st.session_state.patient_name and st.session_state.patient_age > 0 and st.session_state.patient_mrn and st.session_state.patient_gender)
    # Update completion status in session state
    st.session_state.patient_info_complete = info_provided
    # Display status message
    if info_provided: st.success("Patient information complete. You can now upload the photo.")
    else: st.warning("Please fill in all patient details above to enable photo upload.")

    # Step 2: File Upload (Conditional)
    st.subheader("2. Upload Patient Photo")
    uploaded_file = st.file_uploader("Upload frontal facial photograph with natural head position",
                                     type=["jpg", "jpeg", "png"],
                                     key="clinical-uploader",
                                     # Disable uploader if info is not complete
                                     disabled=not st.session_state.patient_info_complete,
                                     help="Requires Patient Name, Age > 0, MRN, and Gender to be entered above.")

    # Step 3: Process Image and Display Results (if uploaded and info complete)
    if uploaded_file is not None and st.session_state.patient_info_complete:
        st.info("Processing image...")
        try:
            # --- Image Loading and Standardization ---
            image = Image.open(uploaded_file).convert("RGB")
            image_np = standardize_image(np.array(image))
            h, w, _ = image_np.shape # Get image dimensions

            # --- MediaPipe Face Mesh Processing ---
            with mp_face_mesh.FaceMesh(
                static_image_mode=True, # Process static image
                max_num_faces=1,        # Detect only one face
                refine_landmarks=True,  # Get finer landmarks (eyes, lips)
                min_detection_confidence=MIN_FACE_CONFIDENCE
            ) as face_mesh:
                results = face_mesh.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)) # Process BGR image

                # --- Check if Face Detected ---
                if not results.multi_face_landmarks:
                    st.error("CRITICAL: Face Not Detected - Verify Patient Positioning or Image Quality")
                else:
                    # --- Landmark Extraction ---
                    landmarks = results.multi_face_landmarks[0].landmark
                    def get_anatomical_point(index):
                        """Converts normalized landmark coordinates to pixel coordinates."""
                        lm = landmarks[index]
                        return (lm.x * w, lm.y * h)

                    # Define landmark indices (adjust based on MediaPipe version if needed)
                    points = {
                        'Trichion_Est': (10, 0.07), # Estimated based on landmark 10 + offset
                        'Glabella': 8,             # Forehead point
                        'Nasion': 168,             # Nasal bridge depression
                        'Subnasale': 94,            # Point below the nose
                        'Menton': 152,             # Chin bottom
                        'Zygion_Left': 454,        # Left cheekbone (approx)
                        'Zygion_Right': 234        # Right cheekbone (approx)
                    }
                    # Calculate pixel coordinates for each landmark
                    landmark_coords = {}
                    for name, val in points.items():
                        if isinstance(val, tuple): # Handle estimated points (index, offset)
                            base_idx, offset = val
                            base = get_anatomical_point(base_idx)
                            landmark_coords[name] = (base[0], base[1] - h * offset) # Offset upwards
                        else: # Direct landmark index
                            landmark_coords[name] = get_anatomical_point(val)

                    # --- Midline Analysis ---
                    nasion = landmark_coords['Nasion']
                    subnasale = landmark_coords['Subnasale']
                    nx, ny = nasion; sx, sy = subnasale
                    midline_dx = sx - nx; midline_dy = sy - ny
                    # Calculate angle relative to vertical (atan2 handles quadrants correctly)
                    midline_angle = math.degrees(math.atan2(midline_dx, midline_dy)) if midline_dy != 0 else 0
                    midline_status = "Aligned" if abs(midline_angle) < 2.0 else "Deviated" # 2-degree threshold

                    # --- Facial Thirds Analysis ---
                    if 'Trichion_Est' in landmark_coords: trichion = landmark_coords['Trichion_Est']
                    else: trichion = (w/2, 0) # Fallback if estimation fails
                    glabella = landmark_coords['Glabella']
                    menton = landmark_coords['Menton']
                    # Calculate vertical distances
                    upper = abs(glabella[1] - trichion[1])
                    middle = abs(subnasale[1] - glabella[1])
                    lower = abs(menton[1] - subnasale[1])
                    total_vertical = upper + middle + lower
                    # Calculate ratios and check balance (within 5% deviation from ideal 1/3)
                    thirds_ratio = (0,0,0); balanced_thirds = False; ratio_deviation = [100]*3
                    if total_vertical > 0:
                        thirds_ratio = (upper / total_vertical, middle / total_vertical, lower / total_vertical)
                        ideal_ratio = (1/3, 1/3, 1/3)
                        ratio_deviation = [abs(thirds_ratio[i] - ideal_ratio[i]) / ideal_ratio[i] * 100 for i in range(3)]
                        balanced_thirds = all(dev < 5 for dev in ratio_deviation)

                    # --- Middle vs Lower Third Comparison ---
                    ml_balance_status = "N/A"; ml_ratio_text = "N/A"; threshold_ml = 45.0 / 55.0; is_balanced_ml = False
                    if lower > 0: # Avoid division by zero
                        middle_lower_ratio = middle / lower
                        is_balanced_ml = middle_lower_ratio >= threshold_ml
                        ml_balance_status = "Seimbang" if is_balanced_ml else "Tidak Seimbang"
                        ml_ratio_text = f"{middle_lower_ratio:.2f}"
                    else: ml_balance_status = "N/A (Lower Third Zero)"

                    # --- Other Measurements ---
                    measurements = {
                        'Facial Height (N-Me)': abs(menton[1] - nasion[1]),
                        'Bizygomatic Width': math.dist(landmark_coords['Zygion_Left'], landmark_coords['Zygion_Right']),
                        'Midline Angle': midline_angle,
                        'Upper Third': upper, 'Middle Third': middle, 'Lower Third': lower,
                        'Facial Index': None
                    }
                    # Calculate Facial Index (Height/Width * 100)
                    if measurements['Bizygomatic Width'] > 0:
                        measurements['Facial Index'] = (measurements['Facial Height (N-Me)'] / measurements['Bizygomatic Width']) * 100
                    else: measurements['Facial Index'] = 0 # Or handle as N/A
                    # Determine Facial Type based on Index
                    facial_types = {
                        'Brachyfacial': (0.0,   84.9),
                        'Mesofacial':   (85.0,  89.9),
                        'Dolichofacial':(90.0, float('inf'))
                    }
                    current_type = "Undefined"
                    if measurements['Facial Index'] is not None:
                        types_found = [k for k,v in facial_types.items() if v[0] <= measurements['Facial Index'] <= v[1]]
                        current_type = types_found[0] if types_found else "Undefined"

                    # --- Visualization ---
                    output_img = image_np.copy() # Create a copy to draw on
                    colors = {'Landmarks': (0,255,0), 'Midline': (0,0,255), 'Thirds': (255,165,0)} # Define colors

                    # Calculate midline vector properties
                    midline_vertical = abs(midline_dx) < 1e-6
                    midline_horizontal = abs(midline_dy) < 1e-6
                    slope_inv = 0 # Inverse slope (dx/dy)
                    if not midline_vertical and not midline_horizontal: slope_inv = midline_dx / midline_dy

                    # Calculate perpendicular vector to midline (-dy, dx)
                    perp_vec = (-midline_dy, midline_dx)
                    mag = math.sqrt(perp_vec[0]**2 + perp_vec[1]**2)
                    norm_perp_vec = (0, 0) # Normalized perpendicular vector
                    if mag > 1e-6: norm_perp_vec = (perp_vec[0]/mag, perp_vec[1]/mag)
                    else: norm_perp_vec = (1, 0) # Default horizontal if midline is just a point

                    # Draw Extended Midline
                    if midline_vertical: midline_start_pt = (int(nx), 0); midline_end_pt = (int(nx), h)
                    elif midline_horizontal: midline_start_pt = (0, int(ny)); midline_end_pt = (w, int(ny))
                    else: x_at_y0 = nx + (0 - ny) * slope_inv; x_at_yh = nx + (h - ny) * slope_inv; midline_start_pt = (int(x_at_y0), 0); midline_end_pt = (int(x_at_yh), h)
                    cv2.line(output_img, midline_start_pt, midline_end_pt, colors['Midline'], 2)

                    # Draw Facial Thirds Lines Perpendicular & Full Width using clipping
                    thirds_landmarks = {'Trichion': trichion, 'Glabella': glabella, 'Subnasale': subnasale, 'Menton': menton}
                    L_large = max(w, h) * 2.0 # Define a very large length

                    for name, lm_point in thirds_landmarks.items():
                        lmy = lm_point[1] # Y-coordinate of the landmark
                        # Find X on midline corresponding to landmark's Y
                        if midline_vertical: midline_x_at_lmy = nx
                        elif midline_horizontal: midline_x_at_lmy = lm_point[0] # Use landmark's X if midline horizontal
                        else: midline_x_at_lmy = nx + (lmy - ny) * slope_inv
                        center_point = (midline_x_at_lmy, lmy) # Point on midline at landmark's height
                        # Calculate endpoints very far away along the perpendicular vector
                        p1_far_x = center_point[0] - L_large * norm_perp_vec[0]; p1_far_y = center_point[1] - L_large * norm_perp_vec[1]
                        p2_far_x = center_point[0] + L_large * norm_perp_vec[0]; p2_far_y = center_point[1] + L_large * norm_perp_vec[1]
                        # Draw the line - OpenCV will clip it at image boundaries
                        cv2.line(output_img, (int(p1_far_x), int(p1_far_y)), (int(p2_far_x), int(p2_far_y)), colors['Thirds'], 2)

                    # Draw Landmarks on top
                    for name, pt in landmark_coords.items():
                        px, py = int(pt[0]), int(pt[1])
                        if 0 <= px < w and 0 <= py < h: # Check bounds
                            cv2.circle(output_img, (px, py), 6, colors['Landmarks'], -1) # Draw circle
                            cv2.putText(output_img, name.split('_')[0], (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['Landmarks'], 1) # Draw label

                    # --- Display Results ---
                    st.subheader("3. Analysis Results")
                    col1, col2 = st.columns(2) # Create two columns for layout
                    with col1:
                        # Display the image with drawings
                        st.image(output_img, caption="Clinical Analysis with Proportions", use_container_width=True)
                    with col2:
                        # --- Prepare data for display ---
                        facial_index_display = f"{measurements['Facial Index']:.1f}" if measurements['Facial Index'] is not None else "N/A"
                        thirds_balance_text = '✅ Balanced Thirds' if balanced_thirds else '❌ Imbalanced Thirds'
                        ml_balance_color = '#2ecc71' if is_balanced_ml else ('#e74c3c' if lower > 0 else '#6c757d')
                        # Get current patient info from session state
                        current_patient_info = {"name": st.session_state.patient_name, "age": st.session_state.patient_age, "mrn": st.session_state.patient_mrn, "gender": st.session_state.patient_gender}

                        # --- Construct HTML for Diagnostic Summary Card ---
                        report_html = f"""<div class="metric-card"> <h3 style="margin-top: 0; color: #2c3e50; margin-bottom: 1.5rem;">Diagnostic Summary</h3> <h4 style="margin:0 0 0.5rem 0; color: #495057; border-bottom: 1px solid #eee; padding-bottom: 0.5rem;">Patient Information</h4> <div style="display: grid; grid-template-columns: auto 1fr; gap: 0.2rem 1rem; margin-bottom: 1.5rem; font-size: 0.95rem;"> <strong>Name:</strong> <span>{current_patient_info['name']}</span> <strong>Age:</strong> <span>{current_patient_info['age']}</span> <strong>MRN:</strong> <span>{current_patient_info['mrn']}</span> <strong>Gender:</strong> <span>{current_patient_info['gender']}</span> </div> <h4 style="margin:0 0 0.5rem 0; color: #495057; border-bottom: 1px solid #eee; padding-bottom: 0.5rem;">Analysis Results</h4> <div style="display: grid; grid-template-columns: auto 1fr; gap: 0.5rem 1rem; margin-bottom: 1rem; font-size: 0.95rem;"> <strong>Facial Type:</strong> <span>{current_type}</span> <strong>Facial Index:</strong> <span>{facial_index_display}</span> <strong>Midline Status:</strong> <span style="color: {'#2ecc71' if midline_status == 'Aligned' else '#e74c3c'};">{midline_status} ({measurements['Midline Angle']:.1f}°)</span> </div> <div style="margin-bottom: 1rem;"> <strong>Facial Thirds:</strong> <div style="padding-left: 1rem; font-size: 0.95rem;"> <div>Upper: {measurements['Upper Third']:.1f}px ({thirds_ratio[0]*100:.1f}%)</div> <div>Middle: {measurements['Middle Third']:.1f}px ({thirds_ratio[1]*100:.1f}%)</div> <div>Lower: {measurements['Lower Third']:.1f}px ({thirds_ratio[2]*100:.1f}%)</div> <div style="margin-top: 0.3rem; font-weight: 500; color: {'#2ecc71' if balanced_thirds else '#e74c3c'};">{thirds_balance_text} (Ratio vs Ideal 1/3)</div> </div> </div> <div style="margin-bottom: 1.5rem;"> <strong>Middle/Lower Proportion:</strong> <div style="padding-left: 1rem; font-size: 0.95rem;"> <span style="color: {ml_balance_color};">{ml_balance_status}</span> <span style="color: #6c757d; margin-left: 0.5rem;">(Ratio: {ml_ratio_text}, Threshold: ≥ {threshold_ml:.3f})</span> </div> </div> <div style="border-top: 1px solid #eee; padding-top: 1rem; margin-top: 0.5rem;"> <h4 style="margin:0 0 0.5rem 0; color: #495057;">Image Details</h4> <div style="display: grid; grid-template-columns: auto 1fr; gap: 0.2rem 1rem; font-size: 0.95rem;"> <strong>Resolution:</strong> <span>{w}x{h} pixels</span> </div> </div> </div>"""
                        # Render the HTML summary card
                        st.markdown(report_html, unsafe_allow_html=True)

                        # --- TXT Download Button ---
                        # Prepare the text data for download
                        download_data_txt = f"""OrthoMesh Clinical Report
===========================
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Patient Information:
--------------------
Name: {current_patient_info['name']}
Age: {current_patient_info['age']}
MRN: {current_patient_info['mrn']}
Gender: {current_patient_info['gender']}

Analysis Results:
-----------------
Facial Type: {current_type}
Facial Index: {facial_index_display}
Midline Status: {midline_status} ({measurements['Midline Angle']:.1f} degrees)

Facial Thirds:
  Upper: {measurements['Upper Third']:.1f}px ({thirds_ratio[0]*100:.1f}%)
  Middle: {measurements['Middle Third']:.1f}px ({thirds_ratio[1]*100:.1f}%)
  Lower: {measurements['Lower Third']:.1f}px ({thirds_ratio[2]*100:.1f}%)
  

Middle/Lower Proportion:
  Balance Status: {ml_balance_status}
  Ratio (M/L): {ml_ratio_text}
  Threshold: >= {threshold_ml:.3f}

Image Details:
--------------
Resolution: {w}x{h} pixels
"""
                        # Create the download button for TXT
                        st.download_button(
                            label="Download Clinical Report (TXT)",
                            data=download_data_txt, # Text content
                            # Generate a unique filename
                            file_name=f"OrthoMesh_Report_{current_patient_info.get('mrn', 'UnknownMRN')}_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain" # Set MIME type for TXT
                        )

        # Handle exceptions during the main processing block
        except Exception as e:
            st.error(f"An error occurred during image processing or analysis: {e}")
            # st.exception(e) # Uncomment for detailed traceback during development

    # Display message if info is complete but no file is uploaded yet
    elif uploaded_file is None and st.session_state.patient_info_complete:
        st.info("Upload patient frontal photograph to begin analysis", icon="⬆️")

# --- About Page ---
elif page == "About the Research":
    # Static content for the About page
    st.markdown(""" <div class="about-container"> <h2 class="about-title">UNIVERSITAS INDONESIA</h2> <h3 class="research-header">OTOMATISASI ANALISA WAJAH MENGGUNAKAN MEDIAPIPE FACE MESH:<br> ANALISIS AKURASI DAN APLIKASI KLINIS</h3> <p style="font-size: 1.1rem; color: #4a4a4a; margin-bottom: 2rem;">PROPOSAL PENELITIAN</p> <div style="margin-bottom: 2rem;"><p style="margin: 0.5rem 0; font-size: 1.1rem;"><strong>Janet Angelica Djaja</strong><br><span style="color: #6c757d;">2206099626</span></p></div> <div style="margin-bottom: 2rem;"><p style="margin: 0.5rem 0; font-size: 1.1rem;"><strong>Pembimbing:</strong><br>Dr. drg. Maria Purbiati, Sp. Ort (K)<br>Prof. Dr. drg. Krisnawati, Sp. Ort (K)</p></div> <div style="border-top: 1px solid #e6e9ef; padding-top: 1.5rem;"><p style="margin: 0; color: #6c757d;">FAKULTAS KEDOKTERAN GIGI<br>PROGRAM PENDIDIKAN DOKTER GIGI SPESIALIS<br>DEPARTEMEN ORTODONSIA<br>2025</p></div> </div> """, unsafe_allow_html=True)

# --- App Footer ---
st.markdown(""" <div style="text-align: center; color: #6c757d; margin-top: 4rem; font-size: 0.9rem;"> <hr style="margin-bottom: 1rem; border-color: #e6e9ef;"> OrthoMesh Clinical Suite • Research Prototype • Made by Residents, for Residents </div> """, unsafe_allow_html=True)
