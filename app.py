import streamlit as st
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from PIL import Image
import io
import matplotlib.pyplot as plt

st.title("Saliva Glucose Estimator")

# === Calibration data ===
# Your H and S values from microfluidic bubbles, normalized and baseline-corrected
# Glucose in µM
data = pd.DataFrame({
    "Glucose": [25, 50, 75, 100, 125],
    "H_corr": [0.721974, 0.732042, 0.732042, 0.740652, 0.786352],  # example from your processed bubbles
    "S_corr": [0.086809, 0.092608, 0.092608, 0.101584, 0.112602]
})

y = data["Glucose"].values

# Fit models
model_H = LinearRegression().fit(data[["H_corr"]], y)
model_S = LinearRegression().fit(data[["S_corr"]], y)
model_HS = LinearRegression().fit(data[["H_corr", "S_corr"]], y)

# === Baseline from blank saliva ===
H_blank_deg = 215
S_blank_percent = 8.2
H_blank = H_blank_deg / 360.0
S_blank = S_blank_percent / 100.0

# === Image upload ===
uploaded_file = st.file_uploader("Upload a saliva bubble image", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Convert to HSV
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h_mean = np.mean(img_hsv[:,:,0]) / 179.0  # OpenCV HSV H is 0-179
    s_mean = np.mean(img_hsv[:,:,1]) / 255.0

    # Baseline correction
    H_corr_input = max(h_mean - H_blank, 0)
    S_corr_input = max(s_mean - S_blank, 0)

    # Predict glucose using multivariate H+S
    df_HS = pd.DataFrame({"H_corr":[H_corr_input], "S_corr":[S_corr_input]})
    glucose_est = model_HS.predict(df_HS)[0]

    st.write(f"**Estimated Saliva Glucose:** {glucose_est:.1f} µM")

    # Optional: show uploaded image
    st.image(image, caption="Uploaded saliva bubble image", use_column_width=True)
