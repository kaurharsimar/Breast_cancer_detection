"""
app.py
-------
Flask REST API that serves the trained breast cancer classifier.

Endpoints:
    GET  /health           -> simple liveness check
    GET  /features          -> lists the 30 features the model expects, in order
    POST /predict            -> accepts a JSON feature vector and returns a prediction

Run locally:
    python api/app.py
Then, from another terminal:
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d @sample_request.json
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# Allow running this file directly (python api/app.py) by adding the
# project root to the path so "models/" is found relative to project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

app = Flask(__name__)

# Load model artifacts once at startup
model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))


@app.route("/")
def home():
    return """
    <h1>Breast Cancer Detection API</h1>
    <p>API is running successfully.</p>
    <ul>
        <li><a href="/health">Health</a></li>
        <li><a href="/features">Features</a></li>
    </ul>
    """


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/features", methods=["GET"])
def features():
    return jsonify({"n_features": len(feature_names), "feature_names": feature_names})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # Accept either {"features": {...}} or a flat dict of feature: value
    features_dict = payload.get("features", payload)

    missing = [f for f in feature_names if f not in features_dict]
    if missing:
        return jsonify({"error": "Missing required features", "missing": missing}), 400

    try:
        x = pd.DataFrame([[float(features_dict[f]) for f in feature_names]], columns=feature_names)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid feature value: {e}"}), 400

    x_scaled = scaler.transform(x)
    pred = int(model.predict(x_scaled)[0])
    proba = model.predict_proba(x_scaled)[0].tolist() if hasattr(model, "predict_proba") else None

    result = {
        "prediction": "Malignant" if pred == 1 else "Benign",
        "prediction_code": pred,
    }
    if proba is not None:
        result["probability_benign"] = round(proba[0], 4)
        result["probability_malignant"] = round(proba[1], 4)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
