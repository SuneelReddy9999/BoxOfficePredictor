from flask import Flask, render_template, request
import pandas as pd
import os
from ml import predict_movie_success, train_model

app = Flask(__name__)

# Load and train model
dataset_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
model = train_model(dataset_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        movie_name = request.form["movie_name"]
        genre = request.form["genre"]  # Get genre from the form
        budget = float(request.form["budget"])
        marketing_spend = float(request.form["marketing_spend"])
        actor_popularity = float(request.form["actor_popularity"])
        director_rating = float(request.form["director_rating"])

        predicted_collection, predicted_category = predict_movie_success(
            model, movie_name, budget, marketing_spend, actor_popularity, director_rating, genre
        )

        return render_template(
            "result.html", movie=movie_name, genre=genre, collection=predicted_collection, category=predicted_category
        )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
