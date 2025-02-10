from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Check if the model file exists before loading
model_path = "movie_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("movie_model.pkl not found! Please train and save the model.")

# Load the trained model
model = joblib.load(model_path)

# Genre mapping for model input
GENRE_MAPPING = {"Action": 0, "Drama": 1, "Comedy": 2, "Thriller": 3, "Sci-Fi": 4}

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        movie_name = request.form['movie_name']
        genre = request.form['genre']
        budget = float(request.form['budget'])
        marketing_spend = float(request.form['marketing_spend'])
        actor_popularity = int(request.form['actor_popularity'])
        director_rating = int(request.form['director_rating'])

        # Convert genre to numeric
        genre_numeric = GENRE_MAPPING.get(genre, -1)  # Default to -1 if genre not found
        if genre_numeric == -1:
            return "Invalid Genre! Please select a valid genre."

        # Prepare input data for the model
        input_data = pd.DataFrame([[budget, marketing_spend, actor_popularity, director_rating, genre_numeric]],
                                  columns=["Budget (Cr)", "Marketing Spend (Cr)", "Lead Actor Popularity", "Director Rating", "Genre"])
        
        # Predict collection
        predicted_collection = model.predict(input_data)[0]

        # Categorize movie success
        category = classify_success(budget, marketing_spend, predicted_collection)

        return render_template('result.html', 
                               movie=movie_name, 
                               genre=genre, 
                               collection=round(predicted_collection, 2), 
                               category=category)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
