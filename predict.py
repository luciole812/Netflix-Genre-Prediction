import joblib
import re
import string
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder

# --- Load saved models and objects ---
model = joblib.load('models/xgboost.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# --- Load custom stopwords ---
with open('data/stopwords_en.txt', 'r') as f:
    stop_words = set(f.read().splitlines())

# --- Text cleaning function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# --- Prediction example ---
def predict_genre(description, release_year=2020, type_='Movie'):
    # Clean the input text
    cleaned = clean_text(description)

    # Apply TF-IDF vectorization
    X_text = tfidf.transform([cleaned])

    # Create non-text features
    type_movie = 1.0 if type_ == 'Movie' else 0.0
    type_show = 1.0 if type_ == 'TV Show' else 0.0

    # Ensure the same feature order as during training
    X_other = np.array([[release_year, type_movie, type_show]], dtype=np.float32)

    # Convert text and numeric features
    from scipy.sparse import csr_matrix
    X_other_sparse = csr_matrix(X_other)

    # Combine text and numeric features
    X_final = hstack([X_text, X_other_sparse])

    # Make prediction
    pred = model.predict(X_final)[0]
    genre = label_encoder.inverse_transform([pred])[0]

    return genre

example_description = "A young woman discovers a secret society of demon hunters and joins them to protect humanity."
predicted_genre = predict_genre(example_description, release_year=2018, type_='TV Show')
print(f"Genre predicted : {predicted_genre}")

