import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(".."))

from src.data_loader import load_data, preprocess_data, get_train_test_split
from src.model import train_models, evaluate_models

# Load and prepare data
df = load_data()
X, y = preprocess_data(df, scale=True)
X_train, X_test, y_train, y_test = get_train_test_split(X, y)

# Train model
models = train_models(X_train, y_train)
model = models['XGBoost']

# SHAP explainer setup
explainer = shap.Explainer(model, X_train)

# Streamlit UI
st.set_page_config(page_title="Medical Cost Predictor", layout="centered")
st.title("💊 Medical Insurance Cost Prediction ")

# --- Tabs for Prediction and Evaluation ---
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Model Evaluation"])

# ---------------- Tab 1: Prediction ----------------
with tab1:
    # User Inputs
    age = st.slider("Age", 18, 65, 30)
    sex = st.radio("Sex", ['male', 'female'])
    bmi = st.slider("BMI", 15.0, 45.0, 25.0)
    children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
    smoker = st.radio("Smoker", ['yes', 'no'])
    region = st.selectbox("Region", ['southeast', 'southwest', 'northeast', 'northwest'])

    # Derived feature
    bmi_category = (
        'Underweight' if bmi < 18.5 else
        'Normal' if bmi < 25 else
        'Overweight' if bmi < 30 else 'Obese'
    )

    # Prepare input for model
    user_df = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex_male': [1 if sex == 'male' else 0],
        'smoker_yes': [1 if smoker == 'yes' else 0],
        'region_northwest': [1 if region == 'northwest' else 0],
        'region_southeast': [1 if region == 'southeast' else 0],
        'region_southwest': [1 if region == 'southwest' else 0],
        'bmi_category_Normal': [1 if bmi_category == 'Normal' else 0],
        'bmi_category_Overweight': [1 if bmi_category == 'Overweight' else 0],
        'bmi_category_Obese': [1 if bmi_category == 'Obese' else 0],
    })

    # Align with model features
    for col in X.columns:
        if col not in user_df.columns:
            user_df[col] = 0
    user_df = user_df[X.columns]  # Ensure correct order

    # Predict and show
    if st.button("Predict Medical Cost"):
        prediction = model.predict(user_df)[0]
        st.subheader(f"💰 Predicted Medical Cost: ${prediction:,.2f}")

        # SHAP explanation
        shap_value = explainer(user_df)

        st.subheader("📊 Feature Impact (SHAP Waterfall Plot)")

        plt.figure(figsize=(14, 6))
        shap.plots.waterfall(shap_value[0], show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.clf()

# ---------------- Tab 2: Model Evaluation ----------------
with tab2:
    st.header("📊 Model Evaluation Dashboard")
    st.write("Comparing all trained models:")

    # Evaluate all models
    results = evaluate_models(models, X_test, y_test)
    results = results.set_index("Model")
    st.dataframe(results)

    # --- RMSE Bar Plot ---
    st.subheader("RMSE Comparison")

    model_names = results.index.tolist()
    rmse_values = results["RMSE"].tolist()
    best_rmse_idx = np.argmin(rmse_values)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['green' if i == best_rmse_idx else 'skyblue' for i in range(len(model_names))]
    bars = ax.bar(range(len(model_names)), rmse_values, color=colors)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylabel("RMSE")
    st.pyplot(fig)
    plt.clf()

    # --- R² Bar Plot ---
    st.subheader("R² Comparison")

    r2_values = results["R²"].tolist()
    best_r2_idx = np.argmax(r2_values)

    fig_r2, ax_r2 = plt.subplots(figsize=(10, 5))
    colors_r2 = ['green' if i == best_r2_idx else 'skyblue' for i in range(len(model_names))]
    bars_r2 = ax_r2.bar(range(len(model_names)), r2_values, color=colors_r2)
    ax_r2.set_xticks(range(len(model_names)))
    ax_r2.set_xticklabels(model_names, rotation=15)
    ax_r2.set_ylabel("R² Score")
    for i, bar in enumerate(bars_r2):
        ax_r2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f"{r2_values[i]:.2f}", ha='center', fontsize=9)
    st.pyplot(fig_r2)
    plt.clf()

    # --- Global SHAP Importance ---
    st.subheader("Global Feature Importance (SHAP)")
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)

    # Rename x-axis label to just "Mean SHAP Value"
    ax = plt.gca()
    ax.set_xlabel("Mean (SHAP) Value")  # <- change happens here
    st.pyplot(plt.gcf())
    plt.clf()