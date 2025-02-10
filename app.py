from flask import Flask, render_template, request

app = Flask(__name__)

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

    # Dummy prediction formula (Replace with ML model)
    collection = (budget * 2) + (marketing_spend * 1.5) + (actor_popularity * 10) + (director_rating * 8)

    if collection > 100:
        category = "Blockbuster"
    elif collection > 50:
        category = "Hit"
    else:
        category = "Average"

    return render_template('result.html', movie=movie_name, genre=genre, collection=collection, category=category)

if __name__ == '__main__':
    app.run(debug=True)
