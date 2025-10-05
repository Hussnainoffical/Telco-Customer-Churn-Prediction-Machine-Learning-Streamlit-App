lco Customer Churn Prediction: Machine Learning + Streamlit App
📖 Overview

This project predicts customer churn for a telecom company using the Telco Customer Churn dataset.
It combines machine learning models with a Streamlit web app so users can input customer details and instantly get churn predictions.

The project is designed to showcase end-to-end ML development: from data preprocessing and model training to deployment and business insights.

🚀 Features

✅ Exploratory Data Analysis (EDA) with correlation heatmaps, distributions, and scatter plots
✅ Preprocessing Pipelines for numerical & categorical data
✅ SMOTE Oversampling to handle class imbalance
✅ Multiple ML Models – Logistic Regression, Random Forest, XGBoost, and KNN
✅ Model Calibration & Cross-validation for better probability estimates
✅ Performance Evaluation – Accuracy, ROC-AUC, Classification Reports
✅ Model Interpretability – Feature Importance, SHAP, Partial Dependence Plots
✅ Business Insights to reduce churn risks
✅ Streamlit App for real-time customer churn prediction

📂 Project Structure
TelcoChurnPrediction/
│── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset
│── churn_model_training.py                 # Training & evaluation pipeline
│── app.py                                  # Streamlit web app
                                

📊 Model Performance

Logistic Regression → Good baseline, interpretable

Random Forest → High accuracy, interpretable, stable

XGBoost → Best performance with highest ROC-AUC

KNN → Simple, but less effective on high-dimensional categorical features

🖥️ Streamlit App Demo

The web app allows you to:

Input customer details (e.g., gender, tenure, contract type, internet service, monthly charges, etc.)

Get churn prediction (Yes/No)

View probability of churn for decision-making
