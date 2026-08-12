"""
streamlit_app.py
------------------
Loan Approval Prediction - Streamlit web app.
Loads the model saved by train_model.py and lets a user enter applicant
details to get a live loan approval prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------------------------------------------
# Page setup
# --------------------------------------------------------------
st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="centered")

st.title("🏦 Loan Approval Prediction")
st.write(
    "This app predicts whether a loan application is likely to be "
    "**Approved** or **Rejected**, using a Logistic Regression model "
    "trained on the Loan Prediction dataset."
)

# --------------------------------------------------------------
# Load the saved model and supporting files
# --------------------------------------------------------------
MODEL_DIR = "model"

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    metrics = joblib.load(os.path.join(MODEL_DIR, "metrics.pkl"))
    final_model_name = joblib.load(os.path.join(MODEL_DIR, "final_model_name.pkl"))
    return model, scaler, encoders, feature_names, metrics, final_model_name

try:
    model, scaler, encoders, feature_names, metrics, final_model_name = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(
        "Could not load the trained model files. Make sure the 'model' folder "
        "(model.pkl, scaler.pkl, encoders.pkl, feature_names.pkl, metrics.pkl, "
        "final_model_name.pkl) is included in the repository. "
        f"Details: {e}"
    )

# --------------------------------------------------------------
# Section 1: Dataset input
# --------------------------------------------------------------
st.header("1. Dataset")

DATA_PATH = "data/loan_approval_data.csv"

data_source = st.radio(
    "Choose a dataset to preview:",
    ["Use the built-in training dataset", "Upload my own CSV file"],
    horizontal=True
)

df_preview = None

if data_source == "Use the built-in training dataset":
    if os.path.exists(DATA_PATH):
        df_preview = pd.read_csv(DATA_PATH)
        st.success(f"Loaded built-in dataset with {df_preview.shape[0]} rows.")
    else:
        st.warning("Built-in dataset file not found in the 'data' folder.")
else:
    uploaded_file = st.file_uploader("Upload a CSV file (same columns as the training data)", type=["csv"])
    if uploaded_file is not None:
        try:
            df_preview = pd.read_csv(uploaded_file)
            st.success(f"Uploaded dataset with {df_preview.shape[0]} rows.")
        except Exception as e:
            st.error(f"Could not read this file. Please upload a valid CSV file. Details: {e}")

if df_preview is not None:
    with st.expander("Preview dataset"):
        st.dataframe(df_preview.head(10))

# --------------------------------------------------------------
# Section 2: User input form
# --------------------------------------------------------------
st.header("2. Enter Applicant Details")

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    with col2:
        coapplicant_income = st.text_input("Coapplicant Income (e.g. 1500)", value="")
        loan_amount = st.text_input("Loan Amount (in thousands, e.g. 120)", value="")
        loan_term = st.selectbox("Loan Amount Term (in days)", [360, 180, 120, 84, 60, 36, 12])
        credit_history = st.selectbox("Credit History", ["Good (has paid past debts)", "Bad / None"])

    submitted = st.form_submit_button("Predict Loan Approval")

# --------------------------------------------------------------
# Section 3: Validation + Prediction
# --------------------------------------------------------------
if submitted:
    errors = []

    # ---- Validate missing inputs ----
    if coapplicant_income.strip() == "":
        errors.append("Coapplicant Income cannot be empty.")
    if loan_amount.strip() == "":
        errors.append("Loan Amount cannot be empty.")

    # ---- Validate input types (must be numbers) ----
    coapplicant_income_value = None
    loan_amount_value = None

    if coapplicant_income.strip() != "":
        try:
            coapplicant_income_value = float(coapplicant_income)
        except ValueError:
            errors.append("Coapplicant Income must be a number (e.g. 1500 or 0).")

    if loan_amount.strip() != "":
        try:
            loan_amount_value = float(loan_amount)
        except ValueError:
            errors.append("Loan Amount must be a number (e.g. 120).")

    # ---- Validate invalid values (negative numbers etc.) ----
    if coapplicant_income_value is not None and coapplicant_income_value < 0:
        errors.append("Coapplicant Income cannot be negative.")
    if loan_amount_value is not None and loan_amount_value <= 0:
        errors.append("Loan Amount must be greater than 0.")

    if not artifacts_loaded:
        errors.append("The model is not loaded, so a prediction cannot be made right now.")

    # ---- Show errors, or run the prediction ----
    if errors:
        st.error("Please fix the following before predicting:")
        for e in errors:
            st.write(f"- {e}")
    else:
        # Build a single-row DataFrame matching the training columns
        credit_history_value = 1 if credit_history.startswith("Good") else 0

        input_dict = {
            "Gender": [gender],
            "Married": [married],
            "Education": [education],
            "CoapplicantIncome": [coapplicant_income_value],
            "LoanAmount": [loan_amount_value],
            "Loan_Amount_Term": [loan_term],
            "Credit_History": [credit_history_value],
            "Property_Area": [property_area],
        }
        input_df = pd.DataFrame(input_dict)

        # Encode the text columns using the SAME encoders used in training
        try:
            for col in ["Gender", "Married", "Education", "Property_Area"]:
                input_df[col] = encoders[col].transform(input_df[col])
        except Exception as e:
            st.error(f"Could not process one of the input values: {e}")
            st.stop()

        # Keep columns in the exact order the model was trained on
        input_df = input_df[feature_names]

        # Scale the input the same way the training data was scaled
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        # --------------------------------------------------------------
        # Section 4: Prediction result
        # --------------------------------------------------------------
        st.header("3. Prediction Result")

        if prediction == 1:
            st.success(f"✅ Loan likely to be **APPROVED** (confidence: {probability[1]*100:.1f}%)")
        else:
            st.error(f"❌ Loan likely to be **REJECTED** (confidence: {probability[0]*100:.1f}%)")

        st.progress(float(probability[1]))
        st.caption(f"Approval probability: {probability[1]*100:.1f}%  |  Rejection probability: {probability[0]*100:.1f}%")

# --------------------------------------------------------------
# Section 5: Basic charts
# --------------------------------------------------------------
st.header("4. Basic Charts")

if df_preview is not None and "Loan_Status" in df_preview.columns:
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        sns.countplot(x="Loan_Status", data=df_preview, ax=ax1)
        ax1.set_title("Loan Status Count")
        st.pyplot(fig1)

    with chart_col2:
        if "ApplicantIncome" in df_preview.columns:
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            sns.histplot(df_preview["ApplicantIncome"], kde=True, bins=20, ax=ax2)
            ax2.set_title("Applicant Income Distribution")
            st.pyplot(fig2)
else:
    st.info("Load or upload a dataset with a 'Loan_Status' column above to see charts here.")

# --------------------------------------------------------------
# Section 6: Model performance section
# --------------------------------------------------------------
st.header("5. Model Performance")

if artifacts_loaded:
    st.write(f"**Final model used in this app:** {final_model_name}")

    metrics_df = pd.DataFrame(metrics).T.round(4)
    metrics_df = metrics_df.sort_values(by="F1 Score", ascending=False)
    st.dataframe(metrics_df)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    metrics_df[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(kind="bar", ax=ax3)
    ax3.set_title("Model Comparison")
    ax3.set_ylabel("Score")
    ax3.set_ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig3)
else:
    st.info("Model performance data is unavailable because the model files could not be loaded.")

st.divider()
st.caption("Built with Streamlit • Loan Approval Prediction • Logistic Regression model")
