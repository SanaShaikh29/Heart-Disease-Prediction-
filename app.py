import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.title {
    text-align: center;
    color: #d62828;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #555;
    font-size: 17px;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result-high {
    background-color: #ffe5e5;
    border-left: 6px solid #d62828;
    padding: 20px;
    border-radius: 10px;
    color: #9b0000;
    font-size: 22px;
    font-weight: 600;
}

.result-low {
    background-color: #e5f8ed;
    border-left: 6px solid #2a9d5b;
    padding: 20px;
    border-radius: 10px;
    color: #16733b;
    font-size: 22px;
    font-weight: 600;
}

.stButton > button {
    width: 100%;
    background-color: #d62828;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    padding: 12px;
    border: none;
}

.stButton > button:hover {
    background-color: #b71c1c;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL FILES
# =========================================================

model = joblib.load("logistic_regg_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">❤️ Heart Disease Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter your health information to get a machine-learning-based prediction.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("🧑 Personal Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=40
    )

with col2:
    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )


st.subheader("🫀 Heart & Medical Information")

col1, col2 = st.columns(2)

with col1:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=200,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        value=200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1]
    )

with col2:

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=150
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        min_value=0.0,
        max_value=6.0,
        value=1.0,
        step=0.1
    )

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PREDICTION
# =========================================================

st.markdown("### 🔍 Check Prediction")

if st.button("❤️ Predict Heart Disease Risk"):

    # -----------------------------------------------------
    # CREATE RAW INPUT
    # -----------------------------------------------------

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        "Sex_" + sex: 1,

        "ChestPainType_" + chest_pain: 1,

        "RestingECG_" + resting_ecg: 1,

        "ExerciseAngina_" + exercise_angina: 1,

        "ST_Slope_" + st_slope: 1
    }

    # -----------------------------------------------------
    # CREATE DATAFRAME
    # -----------------------------------------------------

    input_df = pd.DataFrame([raw_input])

    # -----------------------------------------------------
    # ADD MISSING COLUMNS
    # -----------------------------------------------------

    for col in expected_columns:

        if col not in input_df.columns:
            input_df[col] = 0

    # -----------------------------------------------------
    # REMOVE EXTRA COLUMNS
    # -----------------------------------------------------

    input_df = input_df[expected_columns]

    # -----------------------------------------------------
    # SCALE INPUT
    # -----------------------------------------------------

    scaled_input = scaler.transform(input_df)

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(scaled_input)[0]

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.markdown("---")

    if prediction == 1:

        st.markdown(
            """
            <div class="result-high">
                ⚠️ Higher Predicted Risk of Heart Disease
            </div>
            """,
            unsafe_allow_html=True
        )

        st.warning(
            "This prediction is generated by a machine-learning model "
            "and should not be considered a medical diagnosis."
        )

    else:

        st.markdown(
            """
            <div class="result-low">
                ✅ Lower Predicted Risk of Heart Disease
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "This prediction is generated by a machine-learning model "
            "and is not a substitute for professional medical advice."
        )s