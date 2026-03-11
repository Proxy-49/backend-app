# app_glucose_bubbles_visual.py

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Saliva Glucose Estimator", layout="wide")
st.title("Glucose Estimator from Microfluidic Bubbles (with Visualization)")

# --------------------------
# RGB to HSV
# --------------------------
def rgb_to_hsv(rgb):
    rgb = np.array(rgb)
    maxc = rgb.max(axis=1)
    minc = rgb.min(axis=1)
    v = maxc
    s = (maxc - minc) / (maxc + 1e-6)
    s[maxc == 0] = 0
    rc = (maxc - rgb[:,0]) / (maxc - minc + 1e-6)
    gc = (maxc - rgb[:,1]) / (maxc - minc + 1e-6)
    bc = (maxc - rgb[:,2]) / (maxc - minc + 1e-6)
    h = np.zeros_like(maxc)
    mask = maxc == rgb[:,0]
    h[mask] = (bc - gc)[mask]
    mask = maxc == rgb[:,1]
    h[mask] = 2.0 + (rc - bc)[mask]
    mask = maxc == rgb[:,2]
    h[mask] = 4.0 + (gc - rc)[mask]
    h = (h / 6.0) % 1.0
    h[minc == maxc] = 0.0
    return np.stack([h, s, v], axis=1)

# --------------------------
# Bubble extraction with visualization
# --------------------------
def extract_bubble_features(image_path, top_n=20, shrink_ratio=0.9):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)

    circles = cv2.HoughCircles(
        img_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=35, minRadius=5, maxRadius=50
    )
    if circles is None:
        raise ValueError("No bubbles detected.")

    circles = np.around(circles).astype(int)
    candidates = []

    for x, y, r in circles[0]:
        r_shrink = int(r * shrink_ratio)
        Y, X = np.ogrid[:img_rgb.shape[0], :img_rgb.shape[1]]
        mask = (X - x)**2 + (Y - y)**2 <= r_shrink**2
        roi_rgb = img_rgb[mask]
        roi_hsv = rgb_to_hsv(roi_rgb / 255.0)
        h_mean, s_mean, v_mean = roi_hsv.mean(axis=0)

        # Relaxed HSV filter for saliva
        if 252/360 <= h_mean <= 290/360 and s_mean >= 0.02 and v_mean >= 0.5:
            score = (h_mean**8) * r_shrink
            candidates.append({
                "x": x, "y": y, "r": r_shrink,
                "roi_hsv": roi_hsv, "score": score
            })

    if len(candidates) == 0:
        raise ValueError("No bubbles passed HSV filter. Try brighter image or adjust thresholds.")

    # Sort by score and select top non-overlapping
    candidates = sorted(candidates, key=lambda b: b["score"], reverse=True)
    selected = []
    for b in candidates:
        overlap = any(np.sqrt((b["x"]-sb["x"])**2 + (b["y"]-sb["y"])**2) < (b["r"] + sb["r"]) for sb in selected)
        if not overlap:
            selected.append(b)
        if len(selected) >= top_n:
            break

    avg_hsv = np.mean([b["roi_hsv"].mean(axis=0) for b in selected], axis=0)

    # Visualization
    img_vis = img_rgb.copy()
    for b in selected:
        cv2.circle(img_vis, (b["x"], b["y"]), b["r"], (255,0,0), 2)

    return avg_hsv, img_rgb, img_vis

# --------------------------
# Calibration (pure glucose) with saliva baseline baked in
# --------------------------
calibration_data = pd.DataFrame({
    "Glucose": [25,50,75,100,125],
    "H": [0.722795, 0.731712, 0.730700, 0.743624, 0.786134],
    "S": [0.086949, 0.093759, 0.097361, 0.107223, 0.121588]
})
H_blank_deg = 12
S_blank_percent = 0.6
H_blank = H_blank_deg / 360.0
S_blank = S_blank_percent / 100.0

calibration_data["H_corr"] = calibration_data["H"] - H_blank
calibration_data["S_corr"] = calibration_data["S"] - S_blank
y_glucose = calibration_data["Glucose"].values

model_H  = LinearRegression().fit(calibration_data[["H_corr"]], y_glucose)
model_S  = LinearRegression().fit(calibration_data[["S_corr"]], y_glucose)
model_HS = LinearRegression().fit(calibration_data[["H_corr","S_corr"]], y_glucose)

# --------------------------
# Streamlit uploader
# --------------------------
uploaded_file = st.file_uploader("Upload saliva bubble image", type=["jpg","png","jpeg"])

if uploaded_file:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_path = f"temp_{timestamp}.jpg"
    with open(temp_path,"wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        avg_hsv, img_rgb, img_vis = extract_bubble_features(temp_path)
        H_avg, S_avg, V_avg = avg_hsv

        # Regression predictions (baseline baked in)
        df_H  = pd.DataFrame({"H_corr":[H_avg]})
        df_S  = pd.DataFrame({"S_corr":[S_avg]})
        df_HS = pd.DataFrame({"H_corr":[H_avg], "S_corr":[S_avg]})

        g_H  = max(model_H.predict(df_H)[0],0)
        g_S  = max(model_S.predict(df_S)[0],0)
        g_HS = max(model_HS.predict(df_HS)[0],0)

        glucose_weighted = 0.5*g_H + 0.3*g_S + 0.2*g_HS

        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
        st.image(img_vis, caption="Detected Bubbles", use_column_width=True)

        st.subheader("Estimated Glucose (µM)")
        st.write(f"H only: {g_H:.1f}")
        st.write(f"S only: {g_S:.1f}")
        st.write(f"H + S multivariate: {g_HS:.1f}")
        st.write(f"Weighted glucose: {glucose_weighted:.1f}")

    except Exception as e:
        st.error(f"Error: {e}")

