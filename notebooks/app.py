import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crop Field Recommendation",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND LABEL ENCODER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "crop_recommendation_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"


try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

except FileNotFoundError as e:

    st.error("❌ Model or Label Encoder file was not found.")

    st.write("Expected files:")

    st.code(
        f"""
{MODEL_PATH}
{ENCODER_PATH}
"""
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🌱 Crop Field Recommendation System")

st.write(
    "Enter the soil and environmental conditions below "
    "to get the most suitable crop recommendation."
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("🌾 Enter Crop Field Conditions")


col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    N = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        max_value=140.0,
        value=61.0,
        step=1.0
    )

    P = st.number_input(
        "Phosphorus (P)",
        min_value=5.0,
        max_value=145.0,
        value=56.0,
        step=1.0
    )

    K = st.number_input(
        "Potassium (K)",
        min_value=5.0,
        max_value=205.0,
        value=56.0,
        step=1.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=8.0,
        max_value=44.0,
        value=29.1,
        step=0.1
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    humidity = st.number_input(
        "Humidity (%)",
        min_value=14.0,
        max_value=100.0,
        value=75.0,
        step=0.1
    )

    ph = st.number_input(
        "Soil pH",
        min_value=3.5,
        max_value=10.0,
        value=7.2,
        step=0.1
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=20.0,
        max_value=300.0,
        value=110.0,
        step=1.0
    )


st.divider()


# ============================================================
# INPUT VALIDATION FUNCTION
# ============================================================

def validate_inputs(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
):

    errors = []

    if N < 0:
        errors.append("Nitrogen cannot be negative.")

    if P < 0:
        errors.append("Phosphorus cannot be negative.")

    if K < 0:
        errors.append("Potassium cannot be negative.")

    if humidity < 0 or humidity > 100:
        errors.append(
            "Humidity must be between 0 and 100%."
        )

    if ph < 0 or ph > 14:
        errors.append(
            "pH must be between 0 and 14."
        )

    if temperature < -50 or temperature > 60:
        errors.append(
            "Temperature value appears unrealistic."
        )

    if rainfall < 0:
        errors.append(
            "Rainfall cannot be negative."
        )

    return errors


# ============================================================
# RECOMMEND CROP BUTTON
# ============================================================

if st.button(
    "🌱 Recommend Crop",
    use_container_width=True
):

    # --------------------------------------------------------
    # VALIDATE INPUTS
    # --------------------------------------------------------

    errors = validate_inputs(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    )


    # --------------------------------------------------------
    # SHOW VALIDATION ERRORS
    # --------------------------------------------------------

    if errors:

        st.error(
            "❌ Please correct the following inputs:"
        )

        for error in errors:
            st.write(f"• {error}")


    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    else:

        # IMPORTANT:
        # Feature order must exactly match training data.

        input_data = pd.DataFrame(
            [[
                N,
                P,
                K,
                temperature,
                humidity,
                ph,
                rainfall
            ]],
            columns=[
                "N",
                "P",
                "K",
                "temperature",
                "humidity",
                "ph",
                "rainfall"
            ]
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)


        # ----------------------------------------------------
        # CONVERT ENCODED LABEL TO CROP NAME
        # ----------------------------------------------------

        crop_name = label_encoder.inverse_transform(
            prediction
        )[0]


        # ----------------------------------------------------
        # PREDICTION PROBABILITIES
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            input_data
        )[0]


        # ----------------------------------------------------
        # TOP 3 PREDICTIONS
        # ----------------------------------------------------

        top_3_indices = probabilities.argsort()[-3:][::-1]

        top_3_crops = label_encoder.inverse_transform(
            top_3_indices
        )

        top_3_probabilities = (
            probabilities[top_3_indices] * 100
        )


        # ----------------------------------------------------
        # MAIN CONFIDENCE
        # ----------------------------------------------------

        confidence = top_3_probabilities[0]


        # ====================================================
        # DISPLAY MAIN RESULT
        # ====================================================

        st.success(
            f"🌱 Recommended Crop: **{crop_name.upper()}**"
        )

        st.info(
            f"🎯 Model Confidence: **{confidence:.2f}%**"
        )


        # ====================================================
        # TOP 3 PREDICTIONS
        # ====================================================

        st.subheader("🔎 Top 3 Crop Predictions")


        result_df = pd.DataFrame(
            {
                "Rank": [1, 2, 3],
                "Crop": [
                    crop.capitalize()
                    for crop in top_3_crops
                ],
                "Probability (%)": [
                    round(probability, 2)
                    for probability in top_3_probabilities
                ]
            }
        )


        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # PROBABILITY BAR CHART
        # ====================================================

        st.subheader("📊 Prediction Probabilities")


        probability_chart = pd.DataFrame(
            {
                "Crop": [
                    crop.capitalize()
                    for crop in top_3_crops
                ],
                "Probability": top_3_probabilities
            }
        )


        st.bar_chart(
            probability_chart.set_index("Crop")
        )


        # ====================================================
        # INPUT CONDITIONS
        # ====================================================

        st.subheader("📋 Input Conditions")


        input_display = pd.DataFrame(
            {
                "Feature": [
                    "Nitrogen (N)",
                    "Phosphorus (P)",
                    "Potassium (K)",
                    "Temperature (°C)",
                    "Humidity (%)",
                    "Soil pH",
                    "Rainfall (mm)"
                ],

                "Value": [
                    N,
                    P,
                    K,
                    temperature,
                    humidity,
                    ph,
                    rainfall
                ]
            }
        )


        st.dataframe(
            input_display,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.divider()

        st.subheader("🤖 Model Information")

        st.write(
            """
            **Model:** Tuned Random Forest Classifier

            **Features:** N, P, K, Temperature, Humidity, pH, Rainfall

            **Feature Scaling:** Not required for Random Forest

            **Outlier Removal:** Not applied because extreme
            agricultural values may contain useful information.

            **Prediction:** Multi-class crop classification
            """
        )