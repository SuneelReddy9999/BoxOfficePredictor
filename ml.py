import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_model(dataset_path):
    df = pd.read_csv(dataset_path)

    # Define Features and Target
    X = df[["Budget (Cr)", "Marketing Spend (Cr)", "Lead Actor Popularity", "Director Rating", "Genre"]]
    y = df["Box Office Collection (Cr)"]

    # One-Hot Encoding for 'Genre'
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["Budget (Cr)", "Marketing Spend (Cr)", "Lead Actor Popularity", "Director Rating"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Genre"])
        ]
    )

    # Define Model Pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42))
    ])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Model
    model.fit(X_train, y_train)

    return model

def classify_success(budget, marketing_spend, collection):
    roi = collection / (budget + marketing_spend)
    if roi >= 5:
        return "Blockbuster"
    elif roi >= 3:
        return "Superhit"
    elif roi >= 2:
        return "Hit"
    elif roi >= 1:
        return "Average"
    else:
        return "Flop"

def predict_movie_success(model, movie_name, budget, marketing_spend, actor_popularity, director_rating, genre):
    input_data = pd.DataFrame([[budget, marketing_spend, actor_popularity, director_rating, genre]],
                              columns=["Budget (Cr)", "Marketing Spend (Cr)", "Lead Actor Popularity", "Director Rating", "Genre"])
    
    prediction = model.predict(input_data)[0]
    category = classify_success(budget, marketing_spend, prediction)
    
    return round(prediction, 2), category
