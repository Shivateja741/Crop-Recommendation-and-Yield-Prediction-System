import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crop Field AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent

RECOMMENDATION_MODEL_PATH = (
    PROJECT_DIR / "notebooks" / "crop_recommendation_model.pkl"
)

LABEL_ENCODER_PATH = (
    PROJECT_DIR / "notebooks" / "label_encoder.pkl"
)

YIELD_MODEL_PATH = (
    CURRENT_DIR / "yield_random_forest_model.pkl"
)

YIELD_PREPROCESSOR_PATH = (
    CURRENT_DIR / "yield_preprocessor.pkl"
)


# ============================================================
# HUGGING FACE CONFIGURATION
# ============================================================

HF_YIELD_REPO_ID = (
    "madarchoot/crop-field-yield-model"
)

HF_YIELD_MODEL_FILENAME = (
    "yield_random_forest_model.pkl"
)


# ============================================================
# VERIFIED DATASET RANGES
# ============================================================

# ------------------------------------------------------------
# Crop Recommendation Dataset
# ------------------------------------------------------------

REC_RANGES = {

    "N": {
        "min": 0.0,
        "max": 140.0,
        "default": 50.0
    },

    "P": {
        "min": 5.0,
        "max": 145.0,
        "default": 50.0
    },

    "K": {
        "min": 5.0,
        "max": 205.0,
        "default": 50.0
    },

    "temperature": {
        "min": 8.825675,
        "max": 43.675493,
        "default": 25.0
    },

    "humidity": {
        "min": 14.258040,
        "max": 99.981876,
        "default": 70.0
    },

    "ph": {
        "min": 3.504752,
        "max": 9.935091,
        "default": 6.5
    },

    "rainfall": {
        "min": 20.211267,
        "max": 298.560117,
        "default": 100.0
    }
}


# ------------------------------------------------------------
# Yield Dataset
# ------------------------------------------------------------

YIELD_RANGES = {

    "Crop_Year": {
        "min": 1997,
        "max": 2020,
        "default": 2020
    },

    "Area": {
        "min": 0.5,
        "max": 50808100.0,
        "default": 9317.0
    },

    "Annual_Rainfall": {
        "min": 301.3,
        "max": 6552.7,
        "default": 1247.6
    },

    "Fertilizer": {
        "min": 54.17,
        "max": 4835406877.0,
        "default": 1234957.44
    },

    "Pesticide": {
        "min": 0.09,
        "max": 15750511.0,
        "default": 2421.9
    },

    "Yield": {
        "min": 0.0,
        "max": 21105.0
    }
}


# ============================================================
# GET YIELD MODEL PATH
# ============================================================

def get_yield_model_path():

    # --------------------------------------------------------
    # LOCAL DEVELOPMENT
    # --------------------------------------------------------

    if YIELD_MODEL_PATH.exists():

        return YIELD_MODEL_PATH

    # --------------------------------------------------------
    # STREAMLIT CLOUD / DEPLOYMENT
    # --------------------------------------------------------

    try:

        downloaded_path = hf_hub_download(
            repo_id=HF_YIELD_REPO_ID,
            filename=HF_YIELD_MODEL_FILENAME
        )

        return Path(downloaded_path)

    except Exception as e:

        raise RuntimeError(
            "Could not download the yield model from "
            f"Hugging Face repository: {HF_YIELD_REPO_ID}"
        ) from e


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # --------------------------------------------------------
    # CROP RECOMMENDATION MODEL
    # --------------------------------------------------------

    recommendation_model = joblib.load(
        RECOMMENDATION_MODEL_PATH
    )

    # --------------------------------------------------------
    # LABEL ENCODER
    # --------------------------------------------------------

    label_encoder = joblib.load(
        LABEL_ENCODER_PATH
    )

    # --------------------------------------------------------
    # YIELD MODEL
    # --------------------------------------------------------

    yield_model_path = get_yield_model_path()

    yield_model = joblib.load(
        yield_model_path
    )

    # --------------------------------------------------------
    # YIELD PREPROCESSOR
    # --------------------------------------------------------

    yield_preprocessor = joblib.load(
        YIELD_PREPROCESSOR_PATH
    )

    return (
        recommendation_model,
        label_encoder,
        yield_model,
        yield_preprocessor
    )


# ============================================================
# MODEL LOADING
# ============================================================

try:

    (
        recommendation_model,
        label_encoder,
        yield_model,
        yield_preprocessor
    ) = load_models()

except Exception as e:

    st.error(
        "❌ ML model loading failed."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #AAB0BA;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 14px;
        background-color: #103D29;
        margin-top: 20px;
    }

    .info-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #18324D;
        margin-top: 15px;
    }

    .metric-value {
        font-size: 42px;
        font-weight: 700;
    }

    .small-text {
        color: #AAB0BA;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="font-size:30px;font-weight:700;">
        🌾 Crop Field AI
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        "### Choose Module"
    )

    module = st.radio(
        "",
        [
            "🌱 Crop Recommendation",
            "📊 Yield Prediction"
        ]
    )

    st.divider()

    st.info(
        "AI-powered crop recommendation and "
        "crop yield prediction system."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌾 Crop Field AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Machine Learning based Crop Recommendation and
    Yield Prediction System
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CROP RECOMMENDATION
# ============================================================

if module == "🌱 Crop Recommendation":

    st.header(
        "🌱 Crop Recommendation"
    )

    st.write(
        """
        Enter the soil and environmental conditions.
        The trained classification model will recommend
        the most suitable crop.
        """
    )

    st.info(
        """
        ℹ️ Input limits are based directly on the original
        crop recommendation training dataset. Values outside
        these ranges are not accepted because the model was
        not trained on them.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        nitrogen = st.number_input(
            "🌱 Nitrogen (N)",
            min_value=REC_RANGES["N"]["min"],
            max_value=REC_RANGES["N"]["max"],
            value=REC_RANGES["N"]["default"],
            step=1.0,
            format="%.2f"
        )

        phosphorus = st.number_input(
            "🧪 Phosphorus (P)",
            min_value=REC_RANGES["P"]["min"],
            max_value=REC_RANGES["P"]["max"],
            value=REC_RANGES["P"]["default"],
            step=1.0,
            format="%.2f"
        )

        potassium = st.number_input(
            "🧪 Potassium (K)",
            min_value=REC_RANGES["K"]["min"],
            max_value=REC_RANGES["K"]["max"],
            value=REC_RANGES["K"]["default"],
            step=1.0,
            format="%.2f"
        )

        temperature = st.number_input(
            "🌡️ Temperature (°C)",
            min_value=REC_RANGES["temperature"]["min"],
            max_value=REC_RANGES["temperature"]["max"],
            value=REC_RANGES["temperature"]["default"],
            step=0.1,
            format="%.2f"
        )

    with col2:

        humidity = st.number_input(
            "💧 Humidity (%)",
            min_value=REC_RANGES["humidity"]["min"],
            max_value=REC_RANGES["humidity"]["max"],
            value=REC_RANGES["humidity"]["default"],
            step=0.1,
            format="%.2f"
        )

        soil_ph = st.number_input(
            "🧪 Soil pH",
            min_value=REC_RANGES["ph"]["min"],
            max_value=REC_RANGES["ph"]["max"],
            value=REC_RANGES["ph"]["default"],
            step=0.1,
            format="%.2f"
        )

        rainfall = st.number_input(
            "🌧️ Rainfall (mm)",
            min_value=REC_RANGES["rainfall"]["min"],
            max_value=REC_RANGES["rainfall"]["max"],
            value=REC_RANGES["rainfall"]["default"],
            step=1.0,
            format="%.2f"
        )

    st.divider()

    # --------------------------------------------------------
    # RECOMMEND BUTTON
    # --------------------------------------------------------

    if st.button(
        "🌱 Recommend Crop",
        use_container_width=True
    ):

        try:

            recommendation_input = pd.DataFrame(
                [[
                    nitrogen,
                    phosphorus,
                    potassium,
                    temperature,
                    humidity,
                    soil_ph,
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

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            prediction = (
                recommendation_model.predict(
                    recommendation_input
                )
            )

            predicted_class = prediction[0]

            # ------------------------------------------------
            # DECODE CROP
            # ------------------------------------------------

            try:

                predicted_crop = (
                    label_encoder
                    .inverse_transform(
                        [predicted_class]
                    )[0]
                )

            except Exception:

                predicted_crop = predicted_class

            predicted_crop = str(
                predicted_crop
            ).strip()

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.success(
                f"🌱 Recommended Crop: "
                f"**{predicted_crop.upper()}**"
            )

            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            if hasattr(
                recommendation_model,
                "predict_proba"
            ):

                probabilities = (
                    recommendation_model
                    .predict_proba(
                        recommendation_input
                    )[0]
                )

                classes = (
                    recommendation_model.classes_
                )

                results = []

                for class_value, probability in zip(
                    classes,
                    probabilities
                ):

                    try:

                        crop_name = (
                            label_encoder
                            .inverse_transform(
                                [class_value]
                            )[0]
                        )

                    except Exception:

                        crop_name = class_value

                    results.append(
                        (
                            str(crop_name).strip(),
                            float(probability)
                        )
                    )

                results.sort(
                    key=lambda x: x[1],
                    reverse=True
                )

                top_results = results[:3]

                # ------------------------------------------------
                # TOP 3
                # ------------------------------------------------

                st.subheader(
                    "🔎 Top 3 Crop Predictions"
                )

                result_df = pd.DataFrame(
                    [
                        {
                            "Rank": i + 1,
                            "Crop": crop,
                            "Probability (%)": round(
                                probability * 100,
                                2
                            )
                        }
                        for i, (
                            crop,
                            probability
                        ) in enumerate(top_results)
                    ]
                )

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------
                # TOP PROBABILITY
                # ------------------------------------------------

                top_probability = (
                    top_results[0][1] * 100
                )

                st.subheader(
                    "📌 Top Prediction Probability"
                )

                st.metric(
                    "Highest Predicted Probability",
                    f"{top_probability:.2f}%"
                )

                # ------------------------------------------------
                # EXPLANATION
                # ------------------------------------------------

                if len(top_results) >= 2:

                    second_probability = (
                        top_results[1][1] * 100
                    )

                    probability_gap = (
                        top_probability
                        - second_probability
                    )

                    if probability_gap < 10:

                        explanation = (
                            f"The model selected "
                            f"**{top_results[0][0].upper()}** "
                            f"because it has the highest "
                            f"predicted probability. However, "
                            f"the probability is relatively "
                            f"close to "
                            f"**{top_results[1][0].upper()}** "
                            f"({second_probability:.2f}%), "
                            f"so the model shows some "
                            f"uncertainty."
                        )

                    else:

                        explanation = (
                            f"The model selected "
                            f"**{top_results[0][0].upper()}** "
                            f"because it has the highest "
                            f"predicted probability among "
                            f"the supported crops."
                        )

                    st.info(
                        explanation
                    )

                st.caption(
                    """
                    Note: The displayed probabilities are
                    model probabilities, not a guarantee of
                    real-world crop performance or yield.
                    """
                )

                # ------------------------------------------------
                # BAR CHART
                # ------------------------------------------------

                chart_df = pd.DataFrame(
                    {
                        "Crop": [
                            item[0]
                            for item in top_results
                        ],
                        "Probability (%)": [
                            item[1] * 100
                            for item in top_results
                        ]
                    }
                )

                st.subheader(
                    "📊 Prediction Probabilities"
                )

                st.bar_chart(
                    chart_df.set_index(
                        "Crop"
                    )
                )

            # ------------------------------------------------
            # INPUT SUMMARY
            # ------------------------------------------------

            st.subheader(
                "📋 Input Summary"
            )

            summary_df = pd.DataFrame(
                {
                    "Parameter": [
                        "Nitrogen (N)",
                        "Phosphorus (P)",
                        "Potassium (K)",
                        "Temperature (°C)",
                        "Humidity (%)",
                        "Soil pH",
                        "Rainfall (mm)"
                    ],

                    "Value": [
                        nitrogen,
                        phosphorus,
                        potassium,
                        temperature,
                        humidity,
                        soil_ph,
                        rainfall
                    ]
                }
            )

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(
                "❌ Crop recommendation failed."
            )

            st.exception(e)


# ============================================================
# YIELD PREDICTION
# ============================================================

elif module == "📊 Yield Prediction":

    st.header(
        "📊 Crop Yield Prediction"
    )

    st.write(
        """
        Enter the crop, season, state and agricultural
        conditions to estimate crop yield.
        """
    )

    st.info(
        """
        ℹ️ Numeric limits are taken directly from the original
        crop-yield training dataset. The model does not use
        Production as an input because Production is not
        available before prediction.
        """
    )

    st.divider()

    # ========================================================
    # YEAR + RAINFALL
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        crop_year = st.number_input(
            "📅 Crop Year",
            min_value=YIELD_RANGES[
                "Crop_Year"
            ]["min"],
            max_value=YIELD_RANGES[
                "Crop_Year"
            ]["max"],
            value=YIELD_RANGES[
                "Crop_Year"
            ]["default"],
            step=1
        )

    with col2:

        annual_rainfall = st.number_input(
            "🌧️ Annual Rainfall (mm)",
            min_value=YIELD_RANGES[
                "Annual_Rainfall"
            ]["min"],
            max_value=YIELD_RANGES[
                "Annual_Rainfall"
            ]["max"],
            value=YIELD_RANGES[
                "Annual_Rainfall"
            ]["default"],
            step=10.0,
            format="%.2f"
        )

    # ========================================================
    # GET CATEGORIES FROM PREPROCESSOR
    # ========================================================

    def extract_categories(
        preprocessor,
        column_name
    ):

        try:

            for (
                transformer_name,
                transformer,
                columns
            ) in preprocessor.transformers_:

                if transformer_name == "remainder":
                    continue

                if not hasattr(
                    transformer,
                    "named_steps"
                ):
                    continue

                for step in (
                    transformer
                    .named_steps
                    .values()
                ):

                    if not hasattr(
                        step,
                        "categories_"
                    ):
                        continue

                    feature_names = getattr(
                        step,
                        "feature_names_in_",
                        None
                    )

                    if feature_names is None:
                        continue

                    feature_names = list(
                        feature_names
                    )

                    if column_name not in feature_names:
                        continue

                    index = feature_names.index(
                        column_name
                    )

                    return list(
                        step.categories_[index]
                    )

        except Exception:
            pass

        return None

    # ========================================================
    # DEFAULT CATEGORIES
    # ========================================================

    DEFAULT_CROPS = [

        "Arecanut",
        "Arhar/Tur",
        "Bajra",
        "Banana",
        "Barley",
        "Black pepper",
        "Cardamom",
        "Cashewnut",
        "Castor seed",
        "Coconut",
        "Coriander",
        "Cotton(lint)",
        "Cowpea(Lobia)",
        "Dry chillies",
        "Garlic",
        "Ginger",
        "Gram",
        "Groundnut",
        "Guar seed",
        "Horse-gram",
        "Jowar",
        "Jute",
        "Khesari",
        "Linseed",
        "Maize",
        "Masoor",
        "Mesta",
        "Moong(Green Gram)",
        "Moth",
        "Niger seed",
        "Onion",
        "Other  Rabi pulses",
        "Other Cereals",
        "Other Kharif Pulses",
        "Other Summer Pulses",
        "Peas & beans (Pulses)",
        "Potato",
        "Ragi",
        "Rapeseed &Mustard",
        "Rice",
        "Safflower",
        "Sannhamp",
        "Sesamum",
        "Small millets",
        "Soyabean",
        "Sugarcane",
        "Sunflower",
        "Tapioca",
        "Tobacco",
        "Turmeric",
        "Urad",
        "Wheat"
    ]

    DEFAULT_SEASONS = [

        "Autumn",
        "Kharif",
        "Rabi",
        "Summer",
        "Whole Year",
        "Winter"
    ]

    DEFAULT_STATES = [

        "Andhra Pradesh",
        "Arunachal Pradesh",
        "Assam",
        "Bihar",
        "Chhattisgarh",
        "Delhi",
        "Goa",
        "Gujarat",
        "Haryana",
        "Himachal Pradesh",
        "Jammu and Kashmir",
        "Jharkhand",
        "Karnataka",
        "Kerala",
        "Madhya Pradesh",
        "Maharashtra",
        "Manipur",
        "Meghalaya",
        "Mizoram",
        "Nagaland",
        "Odisha",
        "Puducherry",
        "Punjab",
        "Rajasthan",
        "Sikkim",
        "Tamil Nadu",
        "Telangana",
        "Tripura",
        "Uttar Pradesh",
        "Uttarakhand",
        "West Bengal"
    ]

    # ========================================================
    # LOAD CATEGORIES
    # ========================================================

    crops = (
        extract_categories(
            yield_preprocessor,
            "Crop"
        )
        or DEFAULT_CROPS
    )

    seasons = (
        extract_categories(
            yield_preprocessor,
            "Season"
        )
        or DEFAULT_SEASONS
    )

    states = (
        extract_categories(
            yield_preprocessor,
            "State"
        )
        or DEFAULT_STATES
    )

    # ========================================================
    # DISPLAY MAPPING
    # ========================================================

    def create_display_mapping(
        values
    ):

        mapping = {}

        for value in values:

            actual_value = str(
                value
            )

            display_value = (
                actual_value.strip()
            )

            if display_value not in mapping:

                mapping[
                    display_value
                ] = actual_value

        return mapping

    crop_mapping = (
        create_display_mapping(
            crops
        )
    )

    season_mapping = (
        create_display_mapping(
            seasons
        )
    )

    state_mapping = (
        create_display_mapping(
            states
        )
    )

    # ========================================================
    # CATEGORICAL INPUTS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        crop_display = st.selectbox(
            "🌱 Crop",
            list(
                crop_mapping.keys()
            )
        )

        season_display = st.selectbox(
            "🌦️ Season",
            list(
                season_mapping.keys()
            )
        )

        state_display = st.selectbox(
            "📍 State",
            list(
                state_mapping.keys()
            )
        )

        area = st.number_input(
            "📐 Cultivated Area",
            min_value=YIELD_RANGES[
                "Area"
            ]["min"],
            max_value=YIELD_RANGES[
                "Area"
            ]["max"],
            value=YIELD_RANGES[
                "Area"
            ]["default"],
            step=10.0,
            format="%.2f"
        )

    with col2:

        fertilizer = st.number_input(
            "🧪 Fertilizer Quantity",
            min_value=YIELD_RANGES[
                "Fertilizer"
            ]["min"],
            max_value=YIELD_RANGES[
                "Fertilizer"
            ]["max"],
            value=YIELD_RANGES[
                "Fertilizer"
            ]["default"],
            step=1000.0,
            format="%.2f"
        )

        pesticide = st.number_input(
            "🧴 Pesticide Quantity",
            min_value=YIELD_RANGES[
                "Pesticide"
            ]["min"],
            max_value=YIELD_RANGES[
                "Pesticide"
            ]["max"],
            value=YIELD_RANGES[
                "Pesticide"
            ]["default"],
            step=10.0,
            format="%.2f"
        )

    st.caption(
        """
        Fertilizer and pesticide are entered as quantities
        because the original dataset contains quantity values,
        not product or chemical names. The model therefore
        predicts based on quantity, not on a specific fertilizer
        or pesticide brand.
        """
    )

    st.divider()

    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    if st.button(
        "📊 Estimate Crop Yield",
        use_container_width=True
    ):

        try:

            # ------------------------------------------------
            # ACTUAL VALUES
            # ------------------------------------------------

            crop = crop_mapping[
                crop_display
            ]

            season = season_mapping[
                season_display
            ]

            state = state_mapping[
                state_display
            ]

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            errors = []

            checks = [

                (
                    "Crop year",
                    crop_year,
                    YIELD_RANGES[
                        "Crop_Year"
                    ]["min"],
                    YIELD_RANGES[
                        "Crop_Year"
                    ]["max"],
                    "Crop year is outside "
                    "the training-data range."
                ),

                (
                    "Cultivated area",
                    area,
                    YIELD_RANGES[
                        "Area"
                    ]["min"],
                    YIELD_RANGES[
                        "Area"
                    ]["max"],
                    "Cultivated area is outside "
                    "the training-data range."
                ),

                (
                    "Annual rainfall",
                    annual_rainfall,
                    YIELD_RANGES[
                        "Annual_Rainfall"
                    ]["min"],
                    YIELD_RANGES[
                        "Annual_Rainfall"
                    ]["max"],
                    "Annual rainfall is outside "
                    "the training-data range."
                ),

                (
                    "Fertilizer quantity",
                    fertilizer,
                    YIELD_RANGES[
                        "Fertilizer"
                    ]["min"],
                    YIELD_RANGES[
                        "Fertilizer"
                    ]["max"],
                    "Fertilizer quantity is outside "
                    "the training-data range."
                ),

                (
                    "Pesticide quantity",
                    pesticide,
                    YIELD_RANGES[
                        "Pesticide"
                    ]["min"],
                    YIELD_RANGES[
                        "Pesticide"
                    ]["max"],
                    "Pesticide quantity is outside "
                    "the training-data range."
                )
            ]

            for (
                name,
                value,
                minimum,
                maximum,
                message
            ) in checks:

                if not (
                    minimum
                    <= value
                    <= maximum
                ):

                    errors.append(
                        message
                    )

            # ------------------------------------------------
            # STOP IF INVALID
            # ------------------------------------------------

            if errors:

                for error in errors:

                    st.error(
                        f"❌ {error}"
                    )

                st.stop()

            # ------------------------------------------------
            # CREATE INPUT DATAFRAME
            # ------------------------------------------------

            yield_input = pd.DataFrame(
                {
                    "Crop": [
                        crop
                    ],

                    "Crop_Year": [
                        int(crop_year)
                    ],

                    "Season": [
                        season
                    ],

                    "State": [
                        state
                    ],

                    "Area": [
                        area
                    ],

                    "Annual_Rainfall": [
                        annual_rainfall
                    ],

                    "Fertilizer": [
                        fertilizer
                    ],

                    "Pesticide": [
                        pesticide
                    ]
                }
            )

            # ------------------------------------------------
            # PREPROCESS
            # ------------------------------------------------

            transformed_input = (
                yield_preprocessor.transform(
                    yield_input
                )
            )

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            predicted_yield = (
                yield_model
                .predict(
                    transformed_input
                )[0]
            )

            predicted_yield = float(
                predicted_yield
            )

            # ------------------------------------------------
            # SAFETY
            # ------------------------------------------------

            if not np.isfinite(
                predicted_yield
            ):

                raise ValueError(
                    "The model returned an invalid prediction."
                )

            predicted_yield = max(
                0.0,
                predicted_yield
            )

            # ------------------------------------------------
            # TOTAL OUTPUT
            # ------------------------------------------------

            estimated_total_output = (
                predicted_yield
                * area
            )

            # =================================================
            # RESULT
            # =================================================

            st.success(
                "🌾 Crop Yield Estimation Completed!"
            )

            st.subheader(
                "🌾 Estimated Crop Yield"
            )

            st.markdown(
                f"""
                <div class="result-box">

                    <div style="
                        font-size:18px;
                        color:#AAB0BA;
                    ">
                        Estimated Yield
                    </div>

                    <div class="metric-value">
                        {predicted_yield:,.2f}
                    </div>

                    <div style="
                        font-size:16px;
                        color:#AAB0BA;
                        margin-top:10px;
                    ">
                        Yield units per unit of cultivated area
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            # =================================================
            # TOTAL OUTPUT
            # =================================================

            st.subheader(
                "📦 Estimated Total Output"
            )

            st.metric(
                "Estimated Output",
                f"{estimated_total_output:,.2f}"
            )

            st.caption(
                """
                Calculated as predicted yield × cultivated
                area. The result uses the same yield/area
                units represented by the training dataset.
                """
            )

            # =================================================
            # INTERPRETATION
            # =================================================

            st.info(
                f"""
                **What does this mean?**

                For the selected **{crop_display}** in
                **{state_display}** during
                **{season_display}**, the model estimates
                a yield of approximately
                **{predicted_yield:,.2f} yield units per
                unit of cultivated area**.

                For the entered cultivated area of
                **{area:,.2f}**, the corresponding derived
                total output estimate is approximately
                **{estimated_total_output:,.2f}**.
                """
            )

            # =================================================
            # MODEL INFORMATION
            # =================================================

            st.subheader(
                "🤖 Model Information"
            )

            st.write(
                """
                The estimate is generated using the trained
                Random Forest Regression model and the saved
                preprocessing pipeline.
                """
            )

            st.caption(
                """
                This is a machine-learning estimate based on
                historical training data. Actual agricultural
                yield can vary because of weather, soil quality,
                irrigation, farming practices and other factors
                not represented in the model.
                """
            )

            # =================================================
            # INPUT SUMMARY
            # =================================================

            st.subheader(
                "📋 Input Summary"
            )

            summary_df = pd.DataFrame(
                {
                    "Parameter": [

                        "Crop",
                        "Crop Year",
                        "Season",
                        "State",
                        "Cultivated Area",
                        "Annual Rainfall",
                        "Fertilizer Quantity",
                        "Pesticide Quantity"
                    ],

                    "Value": [

                        crop_display,
                        crop_year,
                        season_display,
                        state_display,
                        area,
                        annual_rainfall,
                        fertilizer,
                        pesticide
                    ]
                }
            )

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(
                "❌ Yield prediction failed."
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Crop Field AI | Machine Learning based "
    "Crop Recommendation & Yield Prediction System"
)


# ============================================================
# YIELD MODEL FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "🔍 Yield Model Feature Importance"
)

try:

    feature_names = (
        yield_preprocessor
        .get_feature_names_out()
    )

    importances = (
        yield_model
        .feature_importances_
    )

    if len(feature_names) == len(
        importances
    ):

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,

                "Importance": importances,

                "Importance (%)":
                    importances * 100
            }
        )

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Feature names and model importances "
            "have different lengths."
        )

except Exception:

    st.warning(
        "Yield feature importance is unavailable "
        "for this saved model."
    )