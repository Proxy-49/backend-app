# app_glucose_only.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from sklearn.linear_model import LinearRegression
from PIL import Image, ImageOps
from streamlit_local_storage import LocalStorage
localS = LocalStorage()
# --------------------------
# Streamlit UI config
# --------------------------
st.set_page_config(page_title="Saliva Glucose Estimator", layout="wide")

st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #eaf8ea;  /* light green background for app */
    color: black !important;    /* all default text black */
}

/* Force all text to black */
html, body, [class*="css"] {
    color: black !important;
}

p, span, label, div, small {
    color: black !important;
}

span[data-baseweb="tag"] {
    background-color: #A5D6A7 !important;  /* active tab green */
    color: black !important;
    border-radius: 4px !important;
    padding: 4px 8px !important;
}

/* Title background */
h1 {
    background-color: #C8E6C9 !important;  /* soft muted gold */
    padding: 10px;
    border-radius: 5px;
}

/* Entire tabs row background */
div[role="tablist"] {
    background-color: #C6EAC6 !important;  /* slightly darker green than background */
    border-radius: 5px;
    padding: 5px;
}

caption, .stCaption, [data-testid="stCaptionContainer"] {
    color: black !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* Tab text color */
button[data-baseweb="tab"] {
    color: black !important;  /* tab text black */
    font-weight: 600;
}

/* Selected tab highlight */
button[data-baseweb="tab"][data-selected="true"] {
    background-color: #A5D6A7 !important;  /* slightly brighter green for active tab */
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Saliva Glucose Monitoring Platform")

# =====================================
# IMAGE STANDARDISATION 
# =====================================
def standardize_image(input_path, output_path):
    img = Image.open(input_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    width, height = img.size
    max_size = 800

    if max(width, height) > max_size:
        if width >= height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    img.save(output_path, format="JPEG", quality=95)
    return output_path

# =====================================
# RGB TO HSV
# =====================================
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

# =====================================
# BUBBLE FEATURE EXTRACTION 
# =====================================
def extract_bubble_features(image_path, top_n=20):

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize intensity 
    img_rgb = img_rgb.astype(np.float32) / 255.0

    # Grayscale
    gray = cv2.cvtColor((img_rgb*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Contrast enhancement
    gray = cv2.equalizeHist(gray)

    # Blur
    gray = cv2.GaussianBlur(gray, (5,5), 1.2)

    # Hough circles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=15,
        param1=80,
        param2=22,
        minRadius=8,
        maxRadius=25
    )

    if circles is None:
        raise ValueError("No bubbles detected")

    circles = np.round(circles[0]).astype(int)

    candidates = []

    for x, y, r in circles:

        r_roi = int(r * 0.95)

        Y, X = np.ogrid[:img_rgb.shape[0], :img_rgb.shape[1]]
        mask = (X - x)**2 + (Y - y)**2 <= r_roi**2

        roi_rgb = img_rgb[mask]

        if roi_rgb.size == 0:
            continue

        roi_hsv = rgb_to_hsv(roi_rgb)

        h_mean, s_mean, v_mean = roi_hsv.mean(axis=0)

        # HSV pink filter
        if (
            250/360 <= h_mean <= 295/360 and
            s_mean >= 0.035 and
            v_mean >= 0.60
        ):
            score = (h_mean**8) * s_mean * v_mean * r_roi

            candidates.append({
                "x": x,
                "y": y,
                "r": r_roi,
                "roi_hsv": roi_hsv,
                "score": score
            })

    if len(candidates) == 0:
        raise ValueError("No valid bubbles detected")

    # =====================================
    # OVERLAP FILTERING
    # =====================================
    candidates = sorted(candidates, key=lambda b: b["score"], reverse=True)

    selected = []

    for b in candidates:
        overlap = False

        for sb in selected:
            dx = b["x"] - sb["x"]
            dy = b["y"] - sb["y"]
            dist = np.sqrt(dx**2 + dy**2)

            if dist < (b["r"] + sb["r"]):
                overlap = True
                break

        if not overlap:
            selected.append(b)

        if len(selected) >= top_n:
            break

    avg_hsv = np.mean(
        [b["roi_hsv"].mean(axis=0) for b in selected],
        axis=0
    )

    return avg_hsv, (img_rgb*255).astype(np.uint8)


# =====================================
# CALIBRATION MODEL
# =====================================
calibration_data = pd.DataFrame({
    "Glucose": [25, 50, 75, 100, 125],
    "H": [0.735825, 0.745060, 0.740868, 0.743964, 0.784736],
    "S": [0.085632, 0.090197, 0.090234, 0.100153, 0.130699]
})

# saliva reference correction
H_blank_deg = 2
S_blank_percent = 0.1

H_blank = H_blank_deg / 360.0
S_blank = S_blank_percent / 100.0

calibration_data["H_corr"] = calibration_data["H"] - H_blank
calibration_data["S_corr"] = calibration_data["S"] - S_blank

y_glucose = calibration_data["Glucose"].values

model_H  = LinearRegression().fit(calibration_data[["H_corr"]], y_glucose)
model_S  = LinearRegression().fit(calibration_data[["S_corr"]], y_glucose)
model_HS = LinearRegression().fit(calibration_data[["H_corr","S_corr"]], y_glucose)

# =====================================
# DEVICE-SPECIFIC HISTORY
# =====================================
if "history" not in st.session_state:
    saved_history = localS.getItem("glucose_history")

    if saved_history is None:
        st.session_state.history = []
    else:
        st.session_state.history = saved_history

# Track last processed upload
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None

# =====================================
# TABS
# =====================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Home",
    "🧪 Saliva Glucose Estimation",
    "📈 History",
    "🍽 Diet",
    "🩸 Finger Prick Guide",
    "❓ Myth Buster"
])

# =====================================
# HOME
# =====================================
with tab1:
    st.header("About Diabetes")

    st.write("""
Diabetes mellitus is a chronic metabolic disorder characterized by elevated glucose levels.

**Type 1 diabetes** results from insufficient insulin production.

**Type 2 diabetes** results from insulin resistance.

This application estimates saliva glucose and hopes to reduce the need for invasive testing.
""")

    st.subheader("Workflow")

    st.subheader("🗺️ How to Use This Web App")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("https://img.icons8.com/fluency/96/test-tube.png")
        st.caption("1️⃣ Go to Saliva Glucose Estimation Tab")
    with col2:
        st.image("https://img.icons8.com/fluency/96/image-file.png")
        st.caption("2️⃣ Upload/Take image")
    with col3:
        st.image("https://img.icons8.com/fluency/96/microscope.png")
        st.caption("3️⃣ Automatic analysis")
    with col4:
        st.image("https://img.icons8.com/fluency/96/combo-chart.png")
        st.caption("4️⃣ View result")


# =====================================
# Saliva Glucose Estimation TAB
# =====================================
with tab2:
    st.header("🧪 Saliva Glucose Estimation")

    meal_state = st.selectbox(
        "Measurement condition",
        ["Fasting", "Post-breakfast", "Post-lunch", "Post-dinner"]
    )

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"]
    )


    # Process only NEW uploads
    if uploaded_file:
        file_id = hash(uploaded_file.getvalue())

        if file_id != st.session_state.last_processed_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = f"temp_{timestamp}.jpg"
            std_path = f"std_{timestamp}.jpg"

            with open(raw_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                standardize_image(raw_path, std_path)
                avg_hsv, img_rgb = extract_bubble_features(std_path)

                H_avg, S_avg, V_avg = avg_hsv

                df_H = pd.DataFrame({"H_corr": [H_avg]})
                df_S = pd.DataFrame({"S_corr": [S_avg]})
                df_HS = pd.DataFrame({
                    "H_corr": [H_avg],
                    "S_corr": [S_avg]
                })

                g_H = max(model_H.predict(df_H)[0], 0)
                g_S = max(model_S.predict(df_S)[0], 0)
                g_HS = max(model_HS.predict(df_HS)[0], 0)

                # Matrix correction
                SALIVA_MATRIX_FACTOR = 2.0
                glucose_raw = 0.2 * g_H + 0.3 * g_S + 0.5 * g_HS
                glucose_weighted = (
                    glucose_raw * SALIVA_MATRIX_FACTOR
                )

                st.image(img_rgb, caption="Uploaded Image")
                st.subheader("Estimated Glucose (µM)")
                st.write(f"**{glucose_weighted:.1f} µM**")

                # Invalid reading check
                if glucose_weighted < 10 or glucose_weighted > 250:
                    st.error(
                        "❌ Glucose reading out of valid range. "
                        "Please re-upload the image."
                    )

                    st.markdown("""
                    <div style="
                        width:100%;
                        background-color:#d9d9d9;
                        border-radius:8px;
                        height:25px;
                    ">
                        <div style="
                            width:100%;
                            background-color:#8b0000;
                            height:25px;
                            border-radius:8px;
                        "></div>
                    </div>
                    <p style="
                        margin-top:6px;
                        font-weight:600;
                        color:black;
                    ">
                        ❌ Invalid reading — image needs re-upload
                    </p>
                    """, unsafe_allow_html=True)

                else:
                    # Risk visualization
                    risk_width = (
                        (glucose_weighted - 10) / (250 - 10)
                    ) * 100

                    risk_width = np.clip(
                        risk_width, 0, 100
                    )

                    if glucose_weighted <= 110:
                        bar_color = "#0b5d1e"
                        risk_text = "🟢 Low Risk"

                    elif glucose_weighted <= 220:
                        bar_color = "#b8860b"
                        risk_text = "🟡 Moderate Risk"

                    else:
                        bar_color = "#8b0000"
                        risk_text = (
                            "🔴 Very High Glucose Level"
                        )

                    st.markdown(f"""
                    <div style="
                        width:100%;
                        background-color:#d9d9d9;
                        border-radius:8px;
                        height:25px;
                    ">
                        <div style="
                            width:{risk_width}%;
                            background-color:{bar_color};
                            height:25px;
                            border-radius:8px;
                        "></div>
                    </div>
                    <p style="
                        margin-top:6px;
                        font-weight:600;
                        color:black;
                    ">
                        {risk_text}
                    </p>
                    """, unsafe_allow_html=True)

                    # Clinical message
                    if 10 <= glucose_weighted <= 110:
                        st.success(
                            "✅ This is within expected healthy "
                            "physiological saliva range."
                        )

                    elif 110 < glucose_weighted <= 220:
                        st.warning(
                            "⚠️ Elevated saliva glucose detected. "
                            "Consider confirmatory finger-prick "
                            "or clinical assessment."
                        )

                    else:
                        st.error(
                            "❌ Very high glucose detected. "
                            "Please confirm with finger-prick "
                            "or clinical assessment."
                        )

                    # Save valid reading
                    
                    sg_time = datetime.now(
                        ZoneInfo("Asia/Singapore")
                    )

                    new_entry = {
                        "Time": sg_time.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "Glucose": round(
                            glucose_weighted, 1
                        ),
                        "MealState": meal_state
                    }

                    st.session_state.history.append(
                        new_entry
                    )

                    localS.setItem(
                            "glucose_history",
                            st.session_state.history
                        )

                    # mark upload as processed
                    st.session_state.last_processed_file = (
                        file_id
                    )


                    # Trend analysis
                    if len(
                        st.session_state.history
                    ) == 1:
                        st.info(
                            "📍 First recorded reading — "
                            "trend will appear from next "
                            "measurement."
                        )

                    else:
                        prev = float(
                            st.session_state.history[-2][
                                "Glucose"
                            ]
                        )

                        delta = (
                            glucose_weighted - prev
                        )

                        if delta > 20:
                            st.warning(
                                f"📈 Rising trend "
                                f"(+{delta:.1f} µM "
                                f"from previous)"
                            )

                        elif delta < -20:
                            st.info(
                                f"📉 Falling trend "
                                f"({delta:.1f} µM "
                                f"from previous)"
                            )

                        else:
                            st.success(
                                "➖ Stable trend"
                            )

            except Exception as e:
                st.error(
                    "No bubbles detected in image, "
                    f"please upload another image. ({e})"
                )
            finally:
                if os.path.exists(raw_path):
                    os.remove(raw_path)
            
                if os.path.exists(std_path):
                    os.remove(std_path)

# ==========================================
#  HISTORY TAB
# ==========================================
with tab3:
    st.header("Historical Results Log")

    # -----------------------------
    # Clear history button
    # -----------------------------
    if st.button("🗑 Clear History"):
        st.session_state.history = []
        localS.deleteItem("glucose_history")
        st.rerun()

    # -----------------------------
    # Display history
    # -----------------------------
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)

        df["Glucose"] = pd.to_numeric(df["Glucose"])
        df["MovingAvg"] = df["Glucose"].rolling(window=3).mean()

        st.dataframe(df)

        st.download_button(
            "⬇ Download History CSV",
            df.to_csv(index=False),
            file_name="glucose_history.csv",
            mime="text/csv"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Average", f"{df['Glucose'].mean():.1f} µM")

        with col2:
            st.metric("Minimum", f"{df['Glucose'].min():.1f} µM")

        with col3:
            st.metric("Maximum", f"{df['Glucose'].max():.1f} µM")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            df["Time"],
            df["Glucose"],
            marker="o",
            label="Glucose"
        )

        ax.plot(
            df["Time"],
            df["MovingAvg"],
            linestyle="--",
            label="Moving Avg (3)"
        )

        ax.axhline(
            df["Glucose"].min(),
            linestyle=":",
            label="Min"
        )

        ax.axhline(
            df["Glucose"].max(),
            linestyle=":",
            label="Max"
        )

        ax.set_ylabel("Glucose (µM)")
        ax.set_xlabel("Date | Time")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()

        st.pyplot(fig)

    else:
        st.info("No historical data available yet.")
# =====================================
# DIET TAB
# =====================================
with tab4:
    st.header("Food Serving Size Estimation")
    st.image("8.png")
    serving_df = pd.DataFrame({
    "Food Group": ["Vegetables", "Protein", "Carbohydrates", "Fats"],
    "Recommended ": ["2-3 Servings", "1-2 Servings", "1 Serving", "1 Serving"]
    })

    st.table(serving_df)

    st.subheader("Carbohydrate Counter")

    carbs_per_serving = {
            # common foods  
            "Rice (1 serving)": 45,
            "Bread (2 slices)": 30,
            "Apple (1 piece)": 15,
            "Banana (1 piece)": 27,
            "Noodles (1 serving)": 40,
            "Pasta (1 serving)": 42,
            "Crackers (5 pieces)": 15,
            "Milk (1 cup)": 12,
            "Chocolate bar (50g)": 25,
            "Glucose tablet (4g)": 4,
            "Orange juice (250ml)": 25,
            "Cereal (1 bowl)": 35,
            "Oats (40g)": 25,
            "Yogurt (150g)": 15,
            "Cookies (3 pieces)": 20,
            
            # Singapore hawker foods
            "Hainanese Chicken Rice (1 serving)": 60,
            "Nasi Lemak (1 serving)": 45,
            "Roti Prata (1 serving)": 35,
            "Laksa (1 serving)": 55,
            "Char Kway Teow (1 serving)": 65,
            "Mee Goreng (1 serving)": 70,
            "Bak Kut Teh (1 serving)": 40,
            "Hokkien Mee (1 serving)": 69,
            "Bee Hoon Goreng (1 serving)": 45,
            "Rojak (1 serving)": 50,
            "Nasi Briyani (1 serving)": 80,
            "Mee Rebus (1 serving)": 60,
            
            # Singapore drinks
            "Teh Tarik (1 cup)": 18,
            "Teh O (1 cup)": 10,
            "Teh C (1 cup)": 15,
            "Kopi (1 cup)": 10,
            "Kopi O (1 cup)": 5,
            "Kopi C (1 cup)": 12,
            "Milo (1 cup)": 25,
            "Horlicks (1 cup)": 30,
            "Bandung (1 cup)": 25,
            "Ice Lemon Tea (1 cup)": 30

            }

    # Multi-select with up to 4 items
    selected_foods = st.multiselect(
        "Select up to 4 food items (leave blank for none)",
        list(carbs_per_serving.keys()),
        max_selections=4
    )

    # Quantity input for each selected food
    total_carbs = 0
    if selected_foods:
        for food in selected_foods:
            qty = st.number_input(
                f"Servings of {food}",
                min_value=0.25, value=1.0, step=0.25,
                key=f"qty_{food}"
            )
            carbs = carbs_per_serving[food] * qty
            total_carbs += carbs
            st.write(f"**{food}:** {carbs:.1f}g carbs")

    st.metric("Total Carbohydrate Intake", f"{total_carbs:.1f} g")


# =====================================
# FINGER PRICK GUIDE
# =====================================
with tab5:
    st.header("Conventional Finger-Prick Monitoring")

    steps = [
        ("1.png", "1. Warming up finger"),
        ("2.png", "2. Pricking finger"),
        ("3.png", "3. Squeezing finger"),
        ("4.png", "4. Measuring blood glucose"),
        ("5.png", "5. Calculating insulin dose"),
        ("6.png", "6. Dialling insulin pen"),
        ("7.png", "7. Delivering insulin")
    ]

    for img, caption in steps:
        st.image(img)
        st.caption(caption)


# =====================================
# MYTH BUSTER TAB
# =====================================
with tab6:
    st.header("Diabetes Myth Buster")

    myths = {
        "Myth 1: Avoid all starchy foods at all cost":
            "All starchy foods break down to glucose, which is the body's preferred source of energy. Choose high-fibre wholegrain options.",

        "Myth 2: All sugar-free products are calorie-free":
            "Sugar-free foods may still contain starch and carbohydrates that raise blood glucose.",

        "Myth 3: People with diabetes should never consume sugar":
            "Sugar may still be consumed in controlled amounts through carbohydrate counting.",

        "Myth 4: Bitter foods lower glucose":
            "Bitter foods do not directly lower blood glucose. Carbohydrate intake is the main driver.",

        "Myth 5: Wholegrain rice means more rice can be eaten":
            "Wholegrain rice has similar carbohydrate content but higher fibre.",

        "Myth 6: Unlimited protein is acceptable":
            "Excess protein still contributes calories and may lead to weight gain.",

        "Myth 7: Favourite sweets should treat hypoglycaemia":
            "Treat low glucose with 15 g fast-acting carbohydrates only.",

        "Myth 8: Fat content does not matter":
            "Fat slows carbohydrate digestion and may prolong elevated glucose.",

        "Myth 9: People with diabetes should not eat fruits":
            "Fresh fruits should still be consumed in controlled portions.",

        "Myth 10: Weight loss cures diabetes":
            "Weight loss improves control but does not necessarily cure diabetes."
    }

    for myth, fact in myths.items():
        with st.expander(myth):
            st.write(f"**FACT:** {fact}")
