from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from scipy.sparse import hstack

app = Flask(__name__)

# Explicit CORS config — allow everything
CORS(app, 
     origins="*",
     allow_headers=["Content-Type"],
     methods=["GET", "POST", "OPTIONS"],
     supports_credentials=False)

# Load model
print("Loading model...")
clf      = joblib.load('classifier.pkl')
char_vec = joblib.load('char_vectorizer.pkl')
word_vec = joblib.load('word_vectorizer.pkl')
print("✅ Model loaded!")

# Handle preflight explicitly
@app.route('/predict', methods=['GET', 'POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400

    text = str(data['text']).strip()
    if not text:
        return jsonify({'error': 'Empty text'}), 400

    Xc = char_vec.transform([text])
    Xw = word_vec.transform([text])
    X  = hstack([Xc, Xw]).tocsr()

    prediction    = int(clf.predict(X)[0])
    probabilities = clf.predict_proba(X)[0].tolist()

    return jsonify({
        'prediction':    prediction,
        'probabilities': probabilities,
        'text':          text
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
