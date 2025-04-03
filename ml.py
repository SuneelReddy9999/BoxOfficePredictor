import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap

def train_models(dataset_path):
    df = pd.read_csv(dataset_path)

    # Features and Target
    X = df[["Budget (Cr)", "Marketing Spend (Cr)", "Lead Actor Popularity", "Director Rating", "Genre"]]
    y = df["Box Office Collection (Cr)"]

    # One-Hot Encoding for 'Genre'
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["Budget (Cr)", "Marketing Spend (Cr)", "Lead Actor Popularity", "Director Rating"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Genre"])
        ]
    )

    # XGBoost Model
    xgb_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42))
    ])

    # Random Forest Model
    rf_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Models
    xgb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Save Models
    joblib.dump(xgb_model, "xgb_model.pkl")
    joblib.dump(rf_model, "rf_model.pkl")

    # SHAP Analysis for Feature Importance (XGBoost)
    explainer = shap.Explainer(xgb_model.named_steps["regressor"])
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    
    print("Models trained and saved successfully!")

train_models("dataset.csv")
