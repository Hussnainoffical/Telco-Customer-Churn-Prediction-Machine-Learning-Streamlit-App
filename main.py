import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, RocCurveDisplay
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib
import shap
from sklearn.inspection import PartialDependenceDisplay

# ===============================
# Load Data
# ===============================
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

X = data.drop(['customerID', 'Churn'], axis=1)
le = LabelEncoder()
y = le.fit_transform(data['Churn'])

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Target distribution:\n", pd.Series(y).value_counts())

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ===============================
# Preprocessing
# ===============================
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# ===============================
# Models
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05, use_label_encoder=False, eval_metric="logloss"),
    "KNN": KNeighborsClassifier(n_neighbors=7)
}

# ===============================
# Evaluate each model
# ===============================
for name, model in models.items():
    print(f"\n=== {name} (Calibrated) ===")

    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=5)

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", calibrated)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # === Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # === ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_proba, name=name)
    plt.show()

    # === 🚀 Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    # === 🚀 Cross-validation for stability
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
    print("Cross-validated ROC AUC:", np.mean(scores))

    # === 🚀 Feature Importance + Interpretability
    if name in ["Random Forest", "XGBoost"]:
        importances = pipeline.named_steps["model"].base_estimator.feature_importances_
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        feat_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        plt.figure(figsize=(10,6))
        sns.barplot(x=feat_importance[:15], y=feat_importance.index[:15])
        plt.title(f"Top 15 Features Driving Churn - {name}")
        plt.show()

        # SHAP Values
        explainer = shap.TreeExplainer(pipeline.named_steps["model"].base_estimator)
        shap_values = explainer.shap_values(preprocessor.fit_transform(X_test))
        shap.summary_plot(shap_values, preprocessor.fit_transform(X_test), feature_names=feature_names)

        # Partial Dependence Plots (Top 3 features)
        PartialDependenceDisplay.from_estimator(
            pipeline.named_steps["model"].base_estimator,
            preprocessor.fit_transform(X_test),
            [0, 1, 2], # first 3 features as example
            feature_names=feature_names
        )
        plt.show()

        # 🚀 Business Insights Example
        print("\n--- Business Insights ---")
        print("1. Customers on month-to-month contracts tend to churn more.")
        print("2. Fiber optic users have higher churn compared to DSL.")
        print("3. Higher monthly charges correlate with higher churn.")

    # 🚀 Deployment (Save Model)
    joblib.dump(pipeline, f"{name.replace(' ', '_').lower()}_churn_model.pkl")
    print(f"Model saved as {name.replace(' ', '_').lower()}_churn_model.pkl")
