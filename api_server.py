"""
api_server.py  —  Flask API for Text Evaluator Chrome Extension
================================================================
Run this on your machine (or any server) to serve model predictions.

Usage:
    pip install flask flask-cors joblib scikit-learn scipy
    python api_server.py

The server will start at http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from scipy.sparse import hstack

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # allow all origins

# ── Load saved model & vectorizers ───────────────────────────────────────────
# Make sure these files exist — run the notebook first to generate them.
print("Loading model and vectorizers...")
clf      = joblib.load('classifier.pkl')
char_vec = joblib.load('char_vectorizer.pkl')
word_vec = joblib.load('word_vectorizer.pkl')
#print("Model loaded successfully!")

# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request body'}), 400

    text = str(data['text']).strip()
    if not text:
        return jsonify({'error': 'Text is empty'}), 400

    # Transform using the same fitted vectorizers
    Xc = char_vec.transform([text])
    Xw = word_vec.transform([text])
    X  = hstack([Xc, Xw]).tocsr()

    # Predict class and probabilities
    prediction    = int(clf.predict(X)[0])
    probabilities = clf.predict_proba(X)[0].tolist()

    return jsonify({
        'prediction':    prediction,
        'probabilities': probabilities,   # list of 3 floats e.g. [0.74, 0.19, 0.07]
        'text':          text
    })

# ── Health check ──────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
