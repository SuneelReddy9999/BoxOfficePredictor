from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("movie_model.pkl")

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

    # Prepare input data for model
    input_data = pd.DataFrame([[budget, marketing_spend, actor_popularity, director_rating, genre]],
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

if __name__ == '__main__':
    app.run(debug=True)
