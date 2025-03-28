import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import math
from collections import deque

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

st.title("Realtime Identifikasi Landmark Wajah dengan Smoothing")

st.markdown("""
Fitur:
- Deteksi landmark wajah: Trichion, Glabella, Subnasale, Menton.
- Menggambar garis horizontal, kotak Bizygomatic, dan extended median line.
- Menghitung deviasi dagu dan jarak facial thirds.
- **Sliding Window Average:** Perhitungan setiap frame diakumulasi dan ditampilkan nilai rata-ratanya (misalnya, dari 30 frame).
""")

# Tombol kontrol kamera
start_button = st.button("Mulai Kamera")
stop_button = st.button("Berhenti Kamera")

# Tempat untuk menampilkan frame video dan teks kalkulasi
frame_placeholder = st.empty()
calc_placeholder = st.empty()

# Buffer untuk smoothing (sliding window average)
window_size = 30  # misalnya 30 frame
upper_thirds = deque(maxlen=window_size)
middle_thirds = deque(maxlen=window_size)
lower_thirds = deque(maxlen=window_size)
median_angles = deque(maxlen=window_size)
chin_angles = deque(maxlen=window_size)
angle_devs = deque(maxlen=window_size)

# Inisialisasi video capture dari webcam
cap = cv2.VideoCapture(0)

if start_button:
    run = True
    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Tidak dapat membaca stream video dari kamera.")
                break

            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            calc_text = ""
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                def get_pixel_coords(idx):
                    lm = landmarks[idx]
                    return (int(lm.x * w), int(lm.y * h))

                # --- Landmark dan Perhitungan ---
                # (1) Trichion: index 10 + offset 7% dari tinggi gambar
                offset_pixels = int(0.07 * h)
                idx10_pt = get_pixel_coords(10)
                tri_x = idx10_pt[0]
                tri_y = idx10_pt[1] - offset_pixels
                if tri_y < 0:
                    tri_y = 0
                trichion_pt = (tri_x, tri_y)
                calc_text += f"Trichion: (x={tri_x}, y={tri_y}) [Offset: {offset_pixels}px]\n"

                # (2) Glabella: index 8
                glabella_pt = get_pixel_coords(8)
                calc_text += f"Glabella: (x={glabella_pt[0]}, y={glabella_pt[1]})\n"

                # (3) Subnasale: index 94
                subnasale_pt = get_pixel_coords(94)
                calc_text += f"Subnasale: (x={subnasale_pt[0]}, y={subnasale_pt[1]})\n"

                # (4) Menton: index 152
                menton_pt = get_pixel_coords(152)
                calc_text += f"Menton: (x={menton_pt[0]}, y={menton_pt[1]})\n"

                # Bizygomatic: Menggunakan landmark 234 (kanan) & 454 (kiri)
                right_bizy_pt = get_pixel_coords(234)
                left_bizy_pt = get_pixel_coords(454)
                bizygomatic_width = np.linalg.norm(np.array(left_bizy_pt) - np.array(right_bizy_pt))
                calc_text += f"Bizygomatic Width: {bizygomatic_width:.1f}px\n"

                # NASION: index 168 (untuk extended median line & kotak bizygomatic)
                nasion_pt = get_pixel_coords(168)
                calc_text += f"Nasion: (x={nasion_pt[0]}, y={nasion_pt[1]})\n"

                # --- Gambar Landmark (Garis Horizontal) ---
                def draw_line_and_label(img, y, label, color):
                    cv2.line(img, (0, y), (w, y), color, 2)
                    cv2.putText(img, label, (10, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                draw_line_and_label(frame, trichion_pt[1], "Trichion", (0, 255, 0))
                draw_line_and_label(frame, glabella_pt[1], "Glabella", (0, 255, 0))
                draw_line_and_label(frame, subnasale_pt[1], "Subnasale", (0, 255, 0))
                draw_line_and_label(frame, menton_pt[1], "Menton", (0, 255, 0))

                # --- Kotak Bizygomatic ---
                left_x = min(left_bizy_pt[0], right_bizy_pt[0])
                right_x = max(left_bizy_pt[0], right_bizy_pt[0])
                top_y = min(nasion_pt[1], menton_pt[1])
                bottom_y = max(nasion_pt[1], menton_pt[1])
                cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (0, 255, 255), 2)
                calc_text += f"Bizygomatic Box: left={left_x}, right={right_x}, top={top_y}, bottom={bottom_y}\n"

                # --- Extended Median Line ---
                dx = subnasale_pt[0] - nasion_pt[0]
                dy = subnasale_pt[1] - nasion_pt[1]
                angle_median = math.atan2(dy, dx)
                angle_median_deg = math.degrees(angle_median)
                calc_text += f"Median Angle: {angle_median_deg:.1f}° (θ = arctan({dy}/{dx}))\n"

                def line_intersections(x0, y0, theta, width, height):
                    candidates = []
                    cos_t = math.cos(theta)
                    sin_t = math.sin(theta)
                    # Sisi kiri
                    if abs(cos_t) > 1e-7:
                        t_left = (0 - x0) / cos_t
                        y_left = y0 + t_left * sin_t
                        if 0 <= y_left <= height:
                            candidates.append((t_left, 'left'))
                    # Sisi kanan
                    if abs(cos_t) > 1e-7:
                        t_right = (width - x0) / cos_t
                        y_right = y0 + t_right * sin_t
                        if 0 <= y_right <= height:
                            candidates.append((t_right, 'right'))
                    # Sisi atas
                    if abs(sin_t) > 1e-7:
                        t_top = (0 - y0) / sin_t
                        x_top = x0 + t_top * cos_t
                        if 0 <= x_top <= width:
                            candidates.append((t_top, 'top'))
                    # Sisi bawah
                    if abs(sin_t) > 1e-7:
                        t_bottom = (height - y0) / sin_t
                        x_bottom = x0 + t_bottom * cos_t
                        if 0 <= x_bottom <= width:
                            candidates.append((t_bottom, 'bottom'))
                    if len(candidates) < 2:
                        return None
                    candidates.sort(key=lambda c: c[0])
                    t1, _ = candidates[0]
                    t2, _ = candidates[-1]
                    x1 = x0 + t1 * cos_t
                    y1 = y0 + t1 * sin_t
                    x2 = x0 + t2 * cos_t
                    y2 = y0 + t2 * sin_t
                    return (int(x1), int(y1)), (int(x2), int(y2))

                line_pts = line_intersections(nasion_pt[0], nasion_pt[1], angle_median, w, h)
                if line_pts:
                    cv2.line(frame, line_pts[0], line_pts[1], (0, 128, 128), 2)
                else:
                    cv2.line(frame, nasion_pt, subnasale_pt, (0, 128, 128), 2)

                # --- Deviasi Dagu ---
                dx_chin = menton_pt[0] - nasion_pt[0]
                dy_chin = menton_pt[1] - nasion_pt[1]
                angle_chin = math.atan2(dy_chin, dx_chin)
                angle_chin_deg = math.degrees(angle_chin)
                angle_dev = angle_chin_deg - angle_median_deg
                if abs(angle_dev) < 5:
                    dev_text = "Garis median dagu hampir lurus."
                elif angle_dev > 0:
                    dev_text = f"Dagu condong ke kanan ({angle_dev:.1f}°)."
                else:
                    dev_text = f"Dagu condong ke kiri ({abs(angle_dev):.1f}°)."
                calc_text += f"Chin Angle: {angle_chin_deg:.1f}° | Deviation: {angle_dev:.1f}° ({dev_text})\n"
                cv2.putText(frame, dev_text, (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # --- Facial Thirds ---
                tri_y_i = trichion_pt[1]
                gla_y_i = glabella_pt[1]
                sub_y_i = subnasale_pt[1]
                men_y_i = menton_pt[1]

                upper_third = abs(gla_y_i - tri_y_i)
                middle_third = abs(sub_y_i - gla_y_i)
                lower_third = abs(men_y_i - sub_y_i)
                total_h = upper_third + middle_third + lower_third

                calc_text += f"Upper Third (Trichion→Glabella): {upper_third} px\n"
                calc_text += f"Middle Third (Glabella→Subnasale): {middle_third} px\n"
                calc_text += f"Lower Third (Subnasale→Menton): {lower_third} px\n"
                calc_text += f"Total Facial Height: {total_h} px\n"

                # Tambahkan nilai tiap perhitungan ke buffer sliding window
                upper_thirds.append(upper_third)
                middle_thirds.append(middle_third)
                lower_thirds.append(lower_third)
                median_angles.append(angle_median_deg)
                chin_angles.append(angle_chin_deg)
                angle_devs.append(angle_dev)

                # Hitung rata-rata dari buffer
                avg_upper = np.mean(upper_thirds)
                avg_middle = np.mean(middle_thirds)
                avg_lower = np.mean(lower_thirds)
                avg_total = avg_upper + avg_middle + avg_lower
                if avg_total > 0:
                    avg_ut_ratio = (avg_upper / avg_total) * 100
                    avg_mt_ratio = (avg_middle / avg_total) * 100
                    avg_lt_ratio = (avg_lower / avg_total) * 100
                else:
                    avg_ut_ratio = avg_mt_ratio = avg_lt_ratio = 0

                avg_median_angle = np.mean(median_angles)
                avg_chin_angle = np.mean(chin_angles)
                avg_angle_dev = np.mean(angle_devs)

                calc_text += "\n--- Rata-rata Sliding Window (30 frame) ---\n"
                calc_text += f"Upper Third: {avg_upper:.1f} px | Middle Third: {avg_middle:.1f} px | Lower Third: {avg_lower:.1f} px\n"
                calc_text += f"Rasio: Upper {avg_ut_ratio:.1f}%, Middle {avg_mt_ratio:.1f}%, Lower {avg_lt_ratio:.1f}%\n"
                calc_text += f"Median Angle: {avg_median_angle:.1f}° | Chin Angle: {avg_chin_angle:.1f}° | Deviation: {avg_angle_dev:.1f}°\n"

            # Tampilkan frame hasil pemrosesan dan perhitungan
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
            calc_placeholder.text(calc_text)

            # Cek jika tombol "Berhenti Kamera" ditekan
            if stop_button:
                run = False
                break

    cap.release()
    st.success("Kamera telah dihentikan.")
else:
    st.info("Klik 'Mulai Kamera' untuk memulai deteksi wajah secara realtime.")
