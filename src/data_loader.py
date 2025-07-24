import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path='../data/raw/insurance.csv'):
    return pd.read_csv(path)

def preprocess_data(df, scale=False):
    df = df.copy()
    
    # BONUS: Create BMI Category
    def bmi_category(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif 18.5 <= bmi < 25:
            return 'Normal'
        elif 25 <= bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    df['bmi_category'] = df['bmi'].apply(bmi_category)
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region', 'bmi_category'], drop_first=True)

    # Split features and target
    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']

    # Scale features (optional but helps with some models)
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

    return X, y

def get_train_test_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)