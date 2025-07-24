import shap
import matplotlib.pyplot as plt
import pandas as pd

def explain_model(model, X_train, X_test, max_display=10):
    # Use TreeExplainer for tree-based models like XGBoost or RandomForest
    explainer = shap.Explainer(model, X_train)

    # Compute SHAP values
    shap_values = explainer(X_test)

    # Summary plot (bar chart for global feature importance)
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=max_display)

    # Summary plot (beeswarm for impact & direction)
    shap.summary_plot(shap_values, X_test, max_display=max_display)

def explain_instance(model, X_test, index=0):
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    # Force plot for a specific prediction
    shap.plots.waterfall(shap_values[index])