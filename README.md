# 💊 Medical Insurance Cost Prediction App

This Streamlit web application predicts individual medical insurance costs based on user input features such as age, BMI, smoking status, and more. It also provides visual explanations using SHAP (SHapley Additive exPlanations) and evaluates multiple regression models to compare their performance.

---

## 📌 Features

- 🎯 Predicts medical insurance charges based on personal details
- 📊 Interactive SHAP waterfall plot for individual prediction explanations
- 🧠 Model comparison (Linear Regression, Ridge, Lasso, Random Forest, XGBoost)
- 📈 Performance metrics: RMSE, MAE, and R²
- 🌍 Global feature importance visualization using SHAP summary plot
- 🧪 Train/Test evaluation using `sklearn` and `xgboost`
- 🛠️ Modular code: structured with separate scripts for data loading, model training, and evaluation
  

---

## 🏁 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/medical-cost-prediction.git
cd medical-cost-prediction

2. Install Dependencies
It's recommended to use a virtual environment:

pip install -r requirements.txt

3. Run the App
streamlit run app.py

📊 Model Evaluation
The app trains and compares the following regression models:
Linear Regression
Ridge Regression
Lasso Regression
Random Forest Regressor
XGBoost Regressor ✅ (used for final predictions)

Metrics used for evaluation:
RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)
R² Score

Bar charts are provided for RMSE and R² across all models.

🧠 SHAP Explanations
🔹 Local Interpretation
SHAP Waterfall Plot: Visual breakdown of individual prediction

🔹 Global Interpretation
SHAP Summary Plot: Mean absolute SHAP value of each feature across the test dataset

📌 Dataset
The dataset used is the classic Medical Cost Personal Dataset. It includes the following features:

age: Age of the individual

sex: Male or female

bmi: Body mass index

children: Number of dependents

smoker: Smoker or not

region: US region

charges: Medical insurance cost (target)

📂 Kaggle Dataset Link

🧪 Example


Add your own screenshots to demonstrate functionality.

📋 Requirements
Install all dependencies via:

pip install -r requirements.txt

Main packages used:
pandas
numpy
scikit-learn
xgboost
matplotlib
shap
streamlit

👨‍💻 Author
Ahmed Abbasi

🌟 Show Your Support
If you found this project useful, feel free to ⭐ the repository!
