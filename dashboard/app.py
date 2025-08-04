import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ----------------------------
# Load Model & Preprocessing
# ----------------------------
model = joblib.load("models/loan_risk_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")

# Determine if model is linear (needs scaling)
is_linear = "LogisticRegression" in str(type(model))

# SHAP explainer
explainer = shap.TreeExplainer(model) if not is_linear else shap.Explainer(model)

# ----------------------------
# UI Config
# ----------------------------
st.set_page_config(page_title="Smart Loan Risk Scoring", page_icon="🏦", layout="wide")

st.title("🏦 Smart Loan Approval & Risk Scoring System")
st.markdown("**Predict loan default risk and get personalized financial advice using Explainable AI (SHAP).**")

# Sidebar - model info
with st.sidebar:
    st.info("ℹ️ **Model Information**")
    st.write(f"**Model:** {type(model).__name__}")
    st.write("**Version:** 1.0.0")
    st.write("**Metrics:** AUC ≈ 0.89 | Accuracy ≈ 0.85")
    st.write("Predicts probability of default & explains key factors with actionable advice.")

# ----------------------------
# Input Section
# ----------------------------
st.header("Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", 18, 100, 30)
    person_income = st.number_input("Annual Income (₹)", 10000, 10000000, 500000)
    loan_amnt = st.number_input("Loan Amount (₹)", 1000, 5000000, 200000)
    loan_percent_income = st.slider("Percent Income (Loan/Income)", 0.0, 1.0, 0.2)
    cb_preson_cred_hist_length = st.number_input("Credit History Length (years)", 0, 50, 5)

with col2:
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])
    loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    cb_person_default_on_file = st.selectbox("Historical Default", ["Y", "N"])

# ----------------------------
# Preprocess Input
# ----------------------------
input_dict = {
    "person_age": person_age,
    "person_income": person_income,
    "loan_amnt": loan_amnt,
    "loan_percent_income": loan_percent_income,
    "cb_preson_cred_hist_length": cb_preson_cred_hist_length,
    "person_home_ownership": person_home_ownership,
    "loan_intent": loan_intent,
    "loan_grade": loan_grade,
    "cb_person_default_on_file": cb_person_default_on_file
}
input_df = pd.DataFrame([input_dict])

# One-hot encode
cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

# Add missing columns
for col in feature_cols:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[feature_cols]

# Scale if linear
input_prepared = scaler.transform(input_encoded) if is_linear else input_encoded

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Risk"):
    probability = model.predict_proba(input_prepared)[0][1]
    percent = probability * 100

    # Determine category
    if percent < 5:
        category = "Low Risk"
        color = "green"
    elif percent < 10:
        category = "Medium Risk"
        color = "yellow"
    else:
        category = "High Risk"
        color = "red"

    # Contextual message (percentile-like)
    context_message = (
        f"Your profile risk is higher than **{int(percent)}%** of applicants — "
        f"{'proceed with caution' if category == 'High Risk' else 'generally safe, but review terms carefully' if category == 'Medium Risk' else 'you’re in a very safe range'}."
    )

    # --- Prediction Result ---
    st.subheader("Prediction Result")
    st.markdown(f"<h3 style='color:{color}'>{category} — Probability: {percent:.2f}%</h3>", unsafe_allow_html=True)
    st.markdown(f"_{context_message}_")

    # --- Gauge Chart ---
    st.subheader("Default Risk Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        title={'text': "Default Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 5], 'color': 'green'},
                {'range': [5, 10], 'color': 'yellow'},
                {'range': [10, 100], 'color': 'red'}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # --- SHAP Local Explanation (Waterfall) ---
    st.subheader("Feature Contribution (Explainability)")
    shap_values = explainer.shap_values(input_encoded if not is_linear else input_prepared)
    if isinstance(shap_values, list):
        shap_value = shap_values[1][0]  # For classifiers
    else:
        shap_value = shap_values[0]

    fig2, ax = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_value,
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
            data=input_encoded.iloc[0, :],
            feature_names=input_encoded.columns
        ),
        show=False
    )
    st.pyplot(fig2)

    # ----------------------------
    # Personalized Advisory Summary
    # ----------------------------
    st.subheader("Advisor’s Recommendation")

    advice_points = []

    if person_income > 400000:
        advice_points.append("Your higher income indicates strong repayment ability.")
    else:
        advice_points.append("Consider increasing income or reducing expenses to strengthen repayment capacity.")

    if loan_amnt > person_income * 0.5:
        advice_points.append("Your requested loan is high relative to income — consider borrowing less or preparing a solid repayment plan.")
    else:
        advice_points.append("Your loan amount seems reasonable compared to your income level.")

    if cb_preson_cred_hist_length < 2:
        advice_points.append("A longer credit history would improve future approvals.")
    else:
        advice_points.append("Your credit history length supports your credibility as a borrower.")

    if cb_person_default_on_file == "Y":
        advice_points.append("Previous defaults raise concerns — maintaining timely payments will help rebuild trust.")
    else:
        advice_points.append("No previous defaults — this is a positive sign for approval.")

    if loan_grade in ["A", "B"]:
        advice_points.append("Your good credit grade strengthens your profile.")
    else:
        advice_points.append("Improving your credit grade would reduce risk and improve approval chances.")

    if category == "High Risk":
        conclusion = "Overall, your profile suggests **higher risk**. Reducing the loan size or improving income/credit habits could increase approval chances."
    elif category == "Medium Risk":
        conclusion = "Overall, your profile suggests **moderate risk**. Proceed cautiously and review repayment terms carefully."
    else:
        conclusion = "Overall, your profile suggests **low risk**. You’re likely to manage this loan comfortably."

    for point in advice_points:
        st.markdown(f"- {point}")

    st.markdown(f"**{conclusion}**")
