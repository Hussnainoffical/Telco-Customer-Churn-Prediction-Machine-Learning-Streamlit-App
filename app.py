import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("logistic_regression_churn_model.pkl")

# Load dataset once just to get all columns
raw_data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
template = raw_data.drop(['customerID', 'Churn'], axis=1).iloc[0:1]  # one row with all columns

st.title("📊 Telco Customer Churn Prediction App")

with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Copy template row
    input_data = template.copy()

    # Update only user-provided fields
    input_data["gender"] = gender
    input_data["SeniorCitizen"] = 1 if senior == "Yes" else 0
    input_data["Partner"] = partner
    input_data["Dependents"] = dependents
    input_data["tenure"] = tenure
    input_data["Contract"] = contract
    input_data["InternetService"] = internet
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges

    # Predict churn
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    churn_label = "Yes" if prediction == 1 else "No"
    st.subheader(f"🔮 Predicted Churn: **{churn_label}**")
    st.write(f"Probability of Churn: {proba:.2f}")
