# @Harshit
# GitHub - harshitdubey0

from flask import Flask, render_template, request
import pandas as pd
import sklearn
import numpy as np
import seaborn as sb
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os

# --- Flask Setup ---
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- NLTK Data Setup ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

# --- Load Model & Vectorizer Safely ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
vector_path = os.path.join(BASE_DIR, "vector.pkl")

try:
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    with open(vector_path, 'rb') as f:
        vector = pickle.load(f)
except Exception as e:
    print(f"❌ Error loading model/vectorizer: {e}")
    loaded_model, vector = None, None


# --- Fake News Detection Function ---
def fake_news_det(news):
    review = re.sub(r'[^a-zA-Z\s]', '', news)
    review = review.lower()
    review = nltk.word_tokenize(review)
    corpus = [lemmatizer.lemmatize(word) for word in review if word not in stpwrds]
    input_data = [' '.join(corpus)]
    vectorized_input_data = vector.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction


# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        message = request.form['news']
        if not message.strip():
            return render_template("prediction.html", prediction_text="⚠️ Please enter some text!")

        if loaded_model is None or vector is None:
            return render_template("prediction.html", prediction_text="⚠️ Model not loaded properly!")

        pred = fake_news_det(message)

        if pred[0] == 1:
            result = "Prediction of the News : Looking Fake News 📰"
        else:
            result = "Prediction of the News : Looking Real News 📰"

        return render_template("prediction.html", prediction_text=result)
    except Exception as e:
        # Show actual error in logs for debugging
        print(f"🔥 Error in /predict route: {e}")
        return render_template("prediction.html", prediction_text="❌ Internal Server Error — check logs")


# --- Main Run (Gunicorn compatible) ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
