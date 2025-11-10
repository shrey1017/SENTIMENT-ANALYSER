from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = request.form['comment']
    comment_vector = vectorizer.transform([comment])
    prediction = model.predict(comment_vector)[0]
    return render_template('index.html', prediction=prediction, comment=comment)

if __name__ == '__main__':
    app.run(debug=True)
# app.py
