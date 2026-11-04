
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

THRESHOLD = 0.30

# ==========================================
# CUSTOM CSS - LIGHT AND DARK MODE
# ==========================================
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: var(--text-color);
        opacity: 0.75;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-card {
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .result-card {
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 24px;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .disclaimer {
        background: var(--secondary-background-color);
        border-left: 4px solid var(--primary-color);
        padding: 16px;
        border-radius: 6px;
        font-size: 0.9rem;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 48px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("diabetes_model.pkl was not found. Please check your project folder.")
    st.stop()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.title("🩺 Diabetes AI")
    st.caption("Machine Learning Prediction System")

    st.divider()

    page = st.radio(
        "Navigation",
        ["Prediction", "About the Project"]
    )

    st.divider()

    st.markdown("### Model Information")
    st.write("**Algorithm:** Random Forest")
    st.write("**Dataset:** Diabetes Clinical Dataset")
    st.write("**Input Features:** 8")
    st.write("**Classification Cutoff:** 0.30 (experimental)")

    st.divider()

    st.caption("Educational machine learning project.")

# ==========================================
# PREDICTION PAGE
# ==========================================
if page == "Prediction":

    st.markdown(
        '<div class="main-title">Diabetes Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Enter the information below to generate a machine learning prediction.</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Complete all fields using the available measurements. "
        "The model will analyse the information and generate a classification."
    )

    # ======================================
    # PERSONAL INFORMATION
    # ======================================
    st.markdown("## Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"],
            help="Select the gender category."
        )

    with col2:
        age = st.number_input(
            "Age (years)",
            min_value=0.08,
            max_value=80.0,
            value=30.0,
            step=1.0,
            help="Enter age in years."
        )

    with col3:
        smoking_history = st.selectbox(
            "Smoking History",
            ["No Info", "never", "not current", "current", "ever", "former"],
            help="Select the applicable smoking history category."
        )

    st.divider()

    # ======================================
    # MEDICAL INFORMATION
    # ======================================
    st.markdown("## Medical Information")

    col4, col5, col6 = st.columns(3)

    with col4:
        hypertension = st.selectbox(
            "Hypertension",
            ["No", "Yes"],
            help="Whether the individual has high blood pressure."
        )

    with col5:
        heart_disease = st.selectbox(
            "Heart Disease",
            ["No", "Yes"],
            help="Whether the individual has a history of heart disease."
        )

    with col6:
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=10.01,
            max_value=95.69,
            value=27.32,
            step=0.1,
            help="Body Mass Index based on weight and height."
        )

    st.divider()

    # ======================================
    # CLINICAL MEASUREMENTS
    # ======================================
    st.markdown("## Clinical Measurements")

    col7, col8 = st.columns(2)

    with col7:
        hba1c = st.number_input(
            "HbA1c Level (%)",
            min_value=3.5,
            max_value=9.0,
            value=5.8,
            step=0.1,
            help="Reflects average blood sugar over approximately 2–3 months."
        )

    with col8:
        glucose = st.number_input(
            "Blood Glucose Level (mg/dL)",
            min_value=80,
            max_value=300,
            value=140,
            step=1,
            help="Blood glucose measurement as represented in the dataset."
        )

    st.divider()

    st.markdown("### Click below to generate a prediction")

    predict_button = st.button(
        "🔍 Predict Diabetes",
        type="primary",
        use_container_width=True
    )

    # ======================================
    # PREDICTION LOGIC
    # ======================================
    if predict_button:

        input_data = pd.DataFrame({
            "gender": [gender],
            "age": [age],
            "hypertension": [1 if hypertension == "Yes" else 0],
            "heart_disease": [1 if heart_disease == "Yes" else 0],
            "smoking_history": [smoking_history],
            "bmi": [bmi],
            "hbA1c_level": [hba1c],
            "blood_glucose_level": [glucose]
        })

        probability = model.predict_proba(input_data)[0][1]
        prediction = int(probability >= THRESHOLD)

        st.divider()
        st.markdown("## 📊 Prediction Result")

        if prediction == 1:

            st.warning(
                "### Diabetes-associated pattern detected\n\n"
                "The model identified a pattern associated with diabetes "
                "in the training dataset."
            )

        else:

            st.success(
                "### No diabetes-associated pattern detected\n\n"
                "The model did not identify a diabetes-associated pattern "
                "in the entered information."
            )

        st.caption(
            "This result is a machine learning classification, not a diagnosis."
        )

        with st.expander("How was this result generated?"):

            st.write(
                "The application uses a Random Forest classification "
                "algorithm trained using a diabetes clinical dataset."
            )

            st.write(
                "Eight input features are processed by the model to "
                "generate its classification."
            )

            st.write(
                "The classification cutoff is set to 0.30 for this "
                "educational project. It is experimental and has not "
                "been clinically validated."
            )

    # ======================================
    # INPUT EXPLANATIONS
    # ======================================
    with st.expander("Understanding the measurements"):

        st.markdown("""
        **BMI (Body Mass Index)**

        A measurement calculated using weight and height.

        **HbA1c Level**

        Measures the percentage of haemoglobin with glucose attached.
        It reflects average blood sugar over approximately 2–3 months.

        **Blood Glucose Level**

        Measures the amount of glucose in the blood at a particular time.

        **Hypertension**

        Refers to high blood pressure.

        **Heart Disease**

        Indicates whether an individual has a history of heart disease.

        **Smoking History**

        Describes the individual's smoking history category.
        """)

# ==========================================
# ABOUT PAGE
# ==========================================
elif page == "About the Project":

    st.title("About the Project")

    st.write(
        "The Diabetes Prediction System is an educational machine "
        "learning application designed to demonstrate how clinical "
        "and personal data can be used for classification."
    )

    st.divider()

    st.subheader("🤖 Machine Learning Model")

    st.write("""
    This project uses a Random Forest Classifier.

    The model was trained using a diabetes clinical dataset containing
    100,000 records before duplicate removal.
    """)

    st.subheader("📋 Input Features")

    features = [
        "Gender",
        "Age",
        "Hypertension",
        "Heart Disease",
        "Smoking History",
        "Body Mass Index (BMI)",
        "HbA1c Level",
        "Blood Glucose Level"
    ]

    for feature in features:
        st.markdown(f"- {feature}")

    st.divider()

    st.subheader("📈 Model Evaluation")

    st.write("**Accuracy:** 96.94%")
    st.write("**ROC-AUC:** 0.9620")
    st.write("**Diabetes Recall at 0.50 cutoff:** 68.9%")

    st.caption(
        "These results are from the held-out test set. "
        "They do not establish clinical validity or real-world diagnostic performance."
    )

    st.divider()

    st.subheader("⚠️ Medical Disclaimer")

    st.markdown("""
    <div class="disclaimer">
    This application is developed for educational and demonstration purposes only.
    The machine learning model is not clinically validated and must not be used
    to diagnose diabetes or replace professional medical advice.
    Consult a qualified healthcare professional for medical assessment.
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.divider()

st.caption(
    "Diabetes Prediction System | Machine Learning Educational Project"
)
