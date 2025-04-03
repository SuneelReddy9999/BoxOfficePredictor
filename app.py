from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load both models
xgb_model = joblib.load("xgb_model.pkl")
rf_model = joblib.load("rf_model.pkl")

# Function to classify movie success
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
    movie_name = request.form['movie_name']
    genre = request.form['genre']
    budget = float(request.form['budget'])
    marketing_spend = float(request.form['marketing_spend'])
    actor_popularity = int(request.form['actor_popularity'])
    director_rating = int(request.form['director_rating'])

    # Prepare input data
    input_data = pd.DataFrame([[budget, marketing_spend, actor_popularity, director_rating, genre]],
                              columns=["Budget (Cr)", "Marketing Spend (Cr)", "Lead Actor Popularity", "Director Rating", "Genre"])

    # Predict collection using both models
    xgb_prediction = xgb_model.predict(input_data)[0]
    rf_prediction = rf_model.predict(input_data)[0]

    # **Hybrid Algorithm** - Weighted Average
    final_prediction = (0.6 * xgb_prediction) + (0.4 * rf_prediction)

    # Categorize movie success
    category = classify_success(budget, marketing_spend, final_prediction)

    # Find the movie poster
    poster_filename = f"static/posters/{movie_name.lower().replace(' ', '')}.jpg"
    if not os.path.exists(poster_filename):
        poster_filename = "static/posters/default.jpg"  # Default poster if not found

    return render_template('result.html', 
                           movie=movie_name, 
                           genre=genre, 
                           collection=round(final_prediction, 2), 
                           category=category,
                           poster=poster_filename)

if __name__ == '__main__':
    app.run(debug=True)
