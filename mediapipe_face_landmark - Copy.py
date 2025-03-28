import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import math

mp_face_mesh = mp.solutions.face_mesh

st.title("OTOMATISASI IDENTIFIKASI LANDMARK WAJAH ORTODONTIK (TRICHION, GLABELLA, SUBNASALE, MENTON) MENGGUNAKAN MEDIAPIPE FACE MESH: ANALISIS AKURASI DAN APLIKASI KLINIS")

st.markdown("""
**Fitur Utama**:
1. **Trichion** (index 10 + offset 7% tinggi).
2. **Glabella** (index 8), **Subnasale** (94), **Menton** (152).
3. **Bizygomatic**: Menggambar kotak (rectangle) dari x-min dan x-max bizygomatic, 
   namun **top** di y Nasion (168) dan **bottom** di y Menton (152).
4. **Garis Median** diperpanjang (extended) melewati batas gambar, mengikuti slope Nasion→Subnasale.
5. **Facial Thirds** & **Deviasi Dagu** (Nasion→Menton vs. Median).
""")

uploaded_file = st.file_uploader("Unggah Gambar Wajah Frontal", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    h, w, _ = image_np.shape

    st.image(image, caption="Gambar Asli", use_column_width=True)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(img_bgr)

        if not results.multi_face_landmarks:
            st.error("Wajah tidak terdeteksi. Pastikan gambar jelas dan frontal.")
        else:
            landmarks = results.multi_face_landmarks[0].landmark

            def get_pixel_coords(idx):
                lm = landmarks[idx]
                return (lm.x * w, lm.y * h)

            # (1) TRICHION
            offset_fraction = 0.07
            offset_pixels = offset_fraction * h
            idx10_pt = get_pixel_coords(10)
            tri_x = idx10_pt[0]
            tri_y = idx10_pt[1] - offset_pixels
            if tri_y < 0:
                tri_y = 0
            trichion_pt = (tri_x, tri_y)

            # (2) GLABELLA (idx8)
            glabella_pt = get_pixel_coords(8)

            # (3) SUBNASALE (idx94)
            subnasale_pt = get_pixel_coords(94)

            # (4) MENTON (idx152)
            menton_pt = get_pixel_coords(152)

            # BIZYGOMATIC: idx234 (kanan), idx454 (kiri)
            right_bizy_pt = get_pixel_coords(234)
            left_bizy_pt = get_pixel_coords(454)
            bizygomatic_width = np.linalg.norm(np.array(left_bizy_pt) - np.array(right_bizy_pt))

            # NASION (idx168)
            nasion_pt = get_pixel_coords(168)

            # ========== Gambar Output =============
            output_img = img_bgr.copy()

            # A. Gambar garis horizontal landmark
            def draw_line_and_label(img, y, label, color):
                cv2.line(img, (0, y), (w, y), color, 2)
                cv2.putText(img, label, (10, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            tri_y_i = int(trichion_pt[1])
            gla_y_i = int(glabella_pt[1])
            sub_y_i = int(subnasale_pt[1])
            men_y_i = int(menton_pt[1])

            draw_line_and_label(output_img, tri_y_i, "Trichion (10+offset)", (0,255,0))
            draw_line_and_label(output_img, gla_y_i, "Glabella (8)", (0,255,0))
            draw_line_and_label(output_img, sub_y_i, "Subnasale (94)", (0,255,0))
            draw_line_and_label(output_img, men_y_i, "Menton (152)", (0,255,0))

            # B. Kotak Bizygomatic: top = y nasion, bottom = y menton
            #    left = min x(bizygomatic left,right), right = max x(bizygomatic left,right)
            nasion_y = nasion_pt[1]
            men_y = menton_pt[1]
            left_x = min(left_bizy_pt[0], right_bizy_pt[0])
            right_x = max(left_bizy_pt[0], right_bizy_pt[0])

            # Pastikan top < bottom
            top_y = min(nasion_y, men_y)
            bottom_y = max(nasion_y, men_y)

            # Gambar rectangle
            cv2.rectangle(
                output_img,
                (int(left_x), int(top_y)),
                (int(right_x), int(bottom_y)),
                (0, 255, 255), 2  # Warna Kuning
            )

            # C. Extended Median Line
            # Garis median = Nasion -> Subnasale
            dx_median = subnasale_pt[0] - nasion_pt[0]
            dy_median = subnasale_pt[1] - nasion_pt[1]
            angle_median = math.atan2(dy_median, dx_median)  # radian

            # Kita tentukan line param:
            # x(t) = x0 + t*cos(angle_median)
            # y(t) = y0 + t*sin(angle_median)
            # Perlu 2 titik: satu di tepi atas/bawah/kiri/kanan, satu lagi di tepi lain

            def line_intersections(x0, y0, theta, width, height):
                """
                Menghitung dua titik potong (x1,y1) & (x2,y2) 
                dari garis melewati (x0,y0) dengan slope theta 
                terhadap bounding box [0,0] -> [width, height].
                """
                # Mencari t untuk 4 kemungkinan (kiri, kanan, atas, bawah).
                # x=0, x=width, y=0, y=height
                # x(t) = x0 + t*cos(theta)
                # y(t) = y0 + t*sin(theta)

                # Buat list (t, boundary) 
                # boundary in { 'top', 'bottom', 'left', 'right' }
                candidates = []

                cos_t = math.cos(theta)
                sin_t = math.sin(theta)

                # 1) x=0 => t = (0 - x0)/cos(theta) jika cos != 0
                if abs(cos_t) > 1e-7:
                    t_left = (0 - x0) / cos_t
                    y_left = y0 + t_left*sin_t
                    if 0 <= y_left <= height:
                        candidates.append((t_left, 'left'))

                # 2) x=width => t = (width - x0)/cos(theta)
                if abs(cos_t) > 1e-7:
                    t_right = (width - x0) / cos_t
                    y_right = y0 + t_right*sin_t
                    if 0 <= y_right <= height:
                        candidates.append((t_right, 'right'))

                # 3) y=0 => t = (0 - y0)/sin(theta) jika sin != 0
                if abs(sin_t) > 1e-7:
                    t_top = (0 - y0) / sin_t
                    x_top = x0 + t_top*cos_t
                    if 0 <= x_top <= width:
                        candidates.append((t_top, 'top'))

                # 4) y=height => t = (height - y0)/sin(theta)
                if abs(sin_t) > 1e-7:
                    t_bottom = (height - y0) / sin_t
                    x_bottom = x0 + t_bottom*cos_t
                    if 0 <= x_bottom <= width:
                        candidates.append((t_bottom, 'bottom'))

                # Kita ambil 2 t yang paling kecil dan paling besar di antara yang valid
                # Agar dapat 2 titik potong di bounding box
                if len(candidates) < 2:
                    # fallback: line doesn't intersect 2 edges? Mungkin miring
                    return None

                candidates.sort(key=lambda c: c[0])
                # Titik paling kecil t dan paling besar t
                t1, _ = candidates[0]
                t2, _ = candidates[-1]

                x1 = x0 + t1*cos_t
                y1 = y0 + t1*sin_t
                x2 = x0 + t2*cos_t
                y2 = y0 + t2*sin_t
                return (x1, y1), (x2, y2)

            nasion_x0, nasion_y0 = nasion_pt
            line_pts = line_intersections(nasion_x0, nasion_y0, angle_median, w, h)
            if line_pts:
                (xm1, ym1), (xm2, ym2) = line_pts
                cv2.line(output_img, (int(xm1), int(ym1)), (int(xm2), int(ym2)), (0,128,128), 2)
            else:
                # fallback: gambar pendek saja
                nasion_i = (int(nasion_x0), int(nasion_y0))
                subnas_i = (int(subnasale_pt[0]), int(subnasale_pt[1]))
                cv2.line(output_img, nasion_i, subnas_i, (0,128,128), 2)

            # D. Deviasi dagu
            angle_median_deg = math.degrees(angle_median)
            dx_chin = menton_pt[0] - nasion_pt[0]
            dy_chin = menton_pt[1] - nasion_pt[1]
            angle_chin_deg = math.degrees(math.atan2(dy_chin, dx_chin))
            angle_dev = angle_chin_deg - angle_median_deg

            if abs(angle_dev) < 5:
                dev_text = "Garis median dagu hampir lurus."
            elif angle_dev > 0:
                dev_text = f"Dagu condong ke kanan ({angle_dev:.1f}°)."
            else:
                dev_text = f"Dagu condong ke kiri ({abs(angle_dev):.1f}°)."

            # Tampilkan final
            output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            st.image(output_rgb, caption="Landmark + Extended Median + Bizygomatic Box & Garis Dagu Vertikal", use_column_width=True)

            # Facial Thirds
            upper_third = abs(gla_y_i - tri_y_i)
            middle_third = abs(sub_y_i - gla_y_i)
            lower_third = abs(men_y_i - sub_y_i)

            st.subheader("Facial Thirds (Vertical Distances)")
            st.write(f"Upper Third (Trichion→Glabella): {upper_third} px")
            st.write(f"Middle Third (Glabella→Subnasale): {middle_third} px")
            st.write(f"Lower Third (Subnasale→Menton): {lower_third} px")

            total_h = upper_third + middle_third + lower_third
            if total_h > 0:
                ut_ratio = (upper_third / total_h) * 100
                mt_ratio = (middle_third / total_h) * 100
                lt_ratio = (lower_third / total_h) * 100
                st.write(f"Upper Third: {ut_ratio:.1f}% | Middle Third: {mt_ratio:.1f}% | Lower Third: {lt_ratio:.1f}%")

            st.subheader("Pengukuran Tambahan")
            st.write(f"Bizygomatic Width: {bizygomatic_width:.1f} px")
            st.write(f"Median Angle (Nasion→Subnasale): {angle_median_deg:.1f}°")
            st.write(f"Chin Angle (Nasion→Menton): {angle_chin_deg:.1f}°")
            st.write(f"Deviation Angle: {angle_dev:.1f}° → {dev_text}")

            st.markdown("""
            ### Interpretasi
            - **Trichion**: Index 10 + offset 7% tinggi.
            - **Glabella (8), Subnasale (94), Menton (152)**: Standar MediaPipe.
            - **Kotak Bizygomatic**: Top = y nasion, Bottom = y menton, Left/Right = x min/max bizygomatic. 
            - **Extended Median Line**: Diperoleh dari slope Nasion→Subnasale, digambar melewati bounding box gambar.
            - **Garis Dagu Vertikal**: Pada x Menton, membentang dari atas ke bawah gambar.
            - **Deviation**: Selisih sudut Nasion→Menton terhadap Nasion→Subnasale. 
              Menunjukkan apakah dagu condong ke kanan/kiri.
            """)
else:
    st.info("Silakan unggah gambar wajah frontal.")
