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
import json
import sqlite3
from datetime import datetime
from io import BytesIO

import joblib
import matplotlib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request, send_file, url_for
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running this file directly (python api/app.py) by adding the
# project root to the path so "models/" is found relative to project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "breast-cancer.csv")
SAMPLE_REQUEST = os.path.join(PROJECT_ROOT, "sample_request.json")
REPORT_PATH = os.path.join(OUTPUTS_DIR, "training_report.json")
DB_PATH = os.path.join(PROJECT_ROOT, "predictions.db")

app = Flask(__name__)

# Load model artifacts once at startup
model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))


def load_training_report():
    try:
        with open(REPORT_PATH, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    drop_cols = [c for c in df.columns if c == "id" or c.lower().startswith("unnamed")]
    return df.drop(columns=drop_cols, errors="ignore")


training_report = load_training_report()
dataset = load_dataset()
feature_ranges = {
    name: {
        "min": float(dataset[name].min()),
        "max": float(dataset[name].max()),
        "mean": float(dataset[name].mean()),
    }
    for name in feature_names
}
dataset_info = {
    "samples": int(dataset.shape[0]),
    "features": len(feature_names),
    "benign": int((dataset["diagnosis"] == "B").sum()),
    "malignant": int((dataset["diagnosis"] == "M").sum()),
    "train_percent": 80,
    "test_percent": 20,
}


def load_sample_features():
    try:
        with open(SAMPLE_REQUEST, "r") as f:
            sample = json.load(f)
        return sample.get("features", {})
    except (OSError, json.JSONDecodeError):
        return {}


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                probability_benign REAL,
                probability_malignant REAL,
                inputs_json TEXT NOT NULL
            )
            """
        )


def validate_features(features_dict):
    errors = {}
    cleaned = {}
    for name in feature_names:
        raw_value = features_dict.get(name, "")
        if raw_value in (None, ""):
            errors[name] = "Required"
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors[name] = "Must be a number"
            continue
        if value < 0:
            errors[name] = "No negative values"
            continue
        cleaned[name] = value
        limits = feature_ranges[name]
        if value < limits["min"] or value > limits["max"]:
            errors[name] = f"Outside dataset range ({limits['min']:.4g} to {limits['max']:.4g})"
    return cleaned, errors


def calculate_top_features(features_dict, limit=8):
    row = pd.DataFrame([[float(features_dict[f]) for f in feature_names]], columns=feature_names)
    scaled_values = scaler.transform(row)[0]
    importances = get_feature_importance_values()
    scores = []
    for name, scaled_value, importance in zip(feature_names, scaled_values, importances):
        scores.append(
            {
                "name": name,
                "score": abs(float(scaled_value)) * max(float(importance), 0.0001),
                "value": float(features_dict[name]),
            }
        )
    scores = sorted(scores, key=lambda item: item["score"], reverse=True)[:limit]
    max_score = max((item["score"] for item in scores), default=1)
    for item in scores:
        item["percent"] = round((item["score"] / max_score) * 100, 1) if max_score else 0
    return scores


def run_prediction(features_dict):
    cleaned, validation_errors = validate_features(features_dict)
    if validation_errors:
        return None, {"error": "Please fix the highlighted input values.", "field_errors": validation_errors}

    x = pd.DataFrame([[cleaned[f] for f in feature_names]], columns=feature_names)
    x_scaled = scaler.transform(x)
    pred = int(model.predict(x_scaled)[0])
    proba = model.predict_proba(x_scaled)[0].tolist() if hasattr(model, "predict_proba") else None

    result = {
        "prediction": "Malignant" if pred == 1 else "Benign",
        "prediction_code": pred,
        "top_features": calculate_top_features(cleaned),
        "inputs": cleaned,
    }
    if proba is not None:
        result["probability_benign"] = round(proba[0], 4)
        result["probability_malignant"] = round(proba[1], 4)
        result["confidence"] = round(max(proba), 4)
    else:
        result["confidence"] = 1.0

    return result, None


def save_prediction(result):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                created_at, prediction, confidence, probability_benign,
                probability_malignant, inputs_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                result["prediction"],
                result["confidence"],
                result.get("probability_benign"),
                result.get("probability_malignant"),
                json.dumps(result["inputs"]),
            ),
        )
        return cursor.lastrowid


def get_prediction_history(limit=10):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, prediction, confidence, probability_benign, probability_malignant
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_prediction_record(prediction_id):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    if row is None:
        return None
    record = dict(row)
    record["inputs"] = json.loads(record["inputs_json"])
    return record


def get_feature_importance_values():
    path = os.path.join(OUTPUTS_DIR, "feature_importance.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            values = json.load(f)
        return [values.get(name, 0.0) for name in feature_names]

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df = dataset.copy()
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    X = df[feature_names]
    y = df["diagnosis"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = scaler.transform(X_test)
    result = permutation_importance(model, X_test_scaled, y_test, n_repeats=8, random_state=42)
    values = {name: float(score) for name, score in zip(feature_names, result.importances_mean)}
    with open(path, "w") as f:
        json.dump(values, f, indent=2)
    return [values.get(name, 0.0) for name in feature_names]


def ensure_dashboard_assets():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    importance_png = os.path.join(OUTPUTS_DIR, "feature_importance.png")
    roc_png = os.path.join(OUTPUTS_DIR, "roc_curve.png")

    importances = get_feature_importance_values()
    top = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)[:10]
    if not os.path.exists(importance_png):
        labels = [name.replace("_", " ").title() for name, _ in top][::-1]
        values = [value for _, value in top][::-1]
        plt.figure(figsize=(9, 5))
        plt.barh(labels, values, color="#0f7c80")
        plt.xlabel("Permutation importance")
        plt.title("Top Feature Importance")
        plt.tight_layout()
        plt.savefig(importance_png, dpi=160)
        plt.close()

    if not os.path.exists(roc_png):
        df = dataset.copy()
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
        X = df[feature_names]
        y = df["diagnosis"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_score = model.predict_proba(scaler.transform(X_test))[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#0f7c80", linewidth=2, label=f"ROC-AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], color="#8a96a8", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_png, dpi=160)
        plt.close()


def asset_url(filename):
    return url_for("output_file", filename=filename)


def pdf_escape(value):
    return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf_report(record):
    lines = [
        "Breast Cancer Detection Report",
        f"Date: {record['created_at']}",
        f"Prediction: {record['prediction']}",
        f"Confidence: {record['confidence'] * 100:.2f}%",
        "Model: Support Vector Machine (RBF)",
        "",
        "Input Values:",
    ]
    for name in feature_names:
        lines.append(f"{name}: {record['inputs'][name]}")

    stream_lines = ["BT", "/F1 11 Tf", "50 790 Td", "14 TL"]
    for line in lines[:48]:
        stream_lines.append(f"({pdf_escape(line)}) Tj")
        stream_lines.append("T*")
    stream_lines.append("ET")
    stream = "\n".join(stream_lines).encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    pdf = BytesIO()
    pdf.write(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(pdf.tell())
        pdf.write(f"{index} 0 obj\n".encode("ascii"))
        pdf.write(obj)
        pdf.write(b"\nendobj\n")
    xref = pdf.tell()
    pdf.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.write(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.write(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode("ascii")
    )
    pdf.seek(0)
    return pdf


init_db()
ensure_dashboard_assets()


def template_context(values=None, result=None, error=None, field_errors=None, prediction_id=None):
    return {
        "feature_names": feature_names,
        "feature_ranges": feature_ranges,
        "values": values or load_sample_features(),
        "result": result,
        "error": error,
        "field_errors": field_errors or {},
        "prediction_id": prediction_id,
        "history": get_prediction_history(),
        "dataset_info": dataset_info,
        "model_comparison": training_report.get("cv_results", {}),
        "confusion_matrix": training_report.get("confusion_matrix", [[0, 0], [0, 0]]),
        "test_metrics": training_report.get("test_metrics", {}),
        "assets": {
            "class_distribution": asset_url("class_distribution.png"),
            "correlation_heatmap": asset_url("correlation_heatmap.png"),
            "top_feature_boxplots": asset_url("top_feature_boxplots.png"),
            "roc_curve": asset_url("roc_curve.png"),
            "feature_importance": asset_url("feature_importance.png"),
        },
    }


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Breast Cancer Detection</title>
    <style>
        :root {
            color-scheme: light;
            --ink: #172033;
            --muted: #627084;
            --line: #d9e0ea;
            --panel: #ffffff;
            --page: #eef3f6;
            --accent: #0f7c80;
            --accent-dark: #0a5f62;
            --danger: #b42318;
            --success: #177245;
            --warn-bg: #fff3e1;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            color: var(--ink);
            background: var(--page);
        }
        header {
            background: #112a3a;
            color: white;
            padding: 28px clamp(16px, 4vw, 48px);
        }
        .header-inner {
            max-width: 1180px;
            margin: 0 auto;
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 18px;
            flex-wrap: wrap;
        }
        h1 {
            margin: 0 0 8px;
            font-size: clamp(28px, 4vw, 44px);
            line-height: 1.05;
            letter-spacing: 0;
        }
        header p {
            margin: 0;
            max-width: 720px;
            color: #c8d6df;
            line-height: 1.55;
        }
        .links {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .links a {
            color: white;
            border: 1px solid rgba(255,255,255,.35);
            border-radius: 6px;
            padding: 9px 12px;
            text-decoration: none;
            font-size: 14px;
        }
        main {
            max-width: 1180px;
            margin: 0 auto;
            padding: 22px clamp(16px, 4vw, 48px) 42px;
        }
        .notice {
            background: var(--warn-bg);
            border: 1px solid #f2cf92;
            border-radius: 6px;
            padding: 12px 14px;
            margin-bottom: 18px;
            color: #66420f;
            line-height: 1.45;
        }
        .layout {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 330px;
            gap: 18px;
            align-items: start;
        }
        form, .side-panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
        }
        .form-head, .panel-section {
            padding: 18px;
            border-bottom: 1px solid var(--line);
        }
        .form-head h2, .panel-section h2 {
            margin: 0 0 6px;
            font-size: 20px;
            letter-spacing: 0;
        }
        .form-head p, .panel-section p {
            margin: 0;
            color: var(--muted);
            line-height: 1.45;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            padding: 18px;
        }
        label {
            display: grid;
            gap: 6px;
            font-size: 13px;
            color: #354156;
            min-width: 0;
        }
        input {
            width: 100%;
            border: 1px solid #c8d2de;
            border-radius: 6px;
            padding: 10px 11px;
            font-size: 15px;
            color: var(--ink);
            background: #fbfcfd;
        }
        input:focus {
            border-color: var(--accent);
            outline: 3px solid rgba(15,124,128,.18);
            background: white;
        }
        .actions {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0 18px 18px;
            flex-wrap: wrap;
        }
        button, .button-link {
            border: 0;
            border-radius: 6px;
            padding: 11px 15px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            text-decoration: none;
        }
        button {
            background: var(--accent);
            color: white;
        }
        button:hover { background: var(--accent-dark); }
        .button-link {
            color: var(--accent-dark);
            background: #e2f3f2;
        }
        .result {
            padding: 18px;
        }
        .result-card {
            border-radius: 8px;
            padding: 16px;
            border: 1px solid var(--line);
            background: #f8fbfc;
        }
        .result-card.malignant {
            border-color: #f0afa9;
            background: #fff1f0;
        }
        .result-card.benign {
            border-color: #acd9c4;
            background: #eefaf4;
        }
        .result-label {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 4px;
        }
        .result-value {
            font-size: 30px;
            font-weight: 800;
            margin-bottom: 12px;
        }
        .malignant .result-value { color: var(--danger); }
        .benign .result-value { color: var(--success); }
        .probability-row {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            border-top: 1px solid rgba(23,32,51,.12);
            padding-top: 10px;
            margin-top: 10px;
            color: #354156;
        }
        .error {
            color: var(--danger);
            background: #fff1f0;
            border: 1px solid #f0afa9;
            border-radius: 6px;
            padding: 12px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            padding: 10px 0;
            border-bottom: 1px solid var(--line);
        }
        .metric:last-child { border-bottom: 0; }
        .metric span:first-child { color: var(--muted); }
        .metric span:last-child { font-weight: 700; }
        @media (max-width: 920px) {
            .layout { grid-template-columns: 1fr; }
            .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 560px) {
            .grid { grid-template-columns: 1fr; }
            .actions { align-items: stretch; }
            button, .button-link { width: 100%; text-align: center; }
        }
    </style>
</head>
<body>
    <header>
        <div class="header-inner">
            <div>
                <h1>Breast Cancer Detection</h1>
                <p>Enter the diagnostic measurements and get the model output directly on this page.</p>
            </div>
            <nav class="links" aria-label="API links">
                <a href="/health">Health</a>
                <a href="/features">Features</a>
            </nav>
        </div>
    </header>
    <main>
        <div class="notice">
            Educational demo only. This prediction is not medical advice or a clinical diagnosis.
        </div>
        <div class="layout">
            <form method="post" action="/">
                <div class="form-head">
                    <h2>Prediction Form</h2>
                    <p>The fields are prefilled with the sample request so you can submit once immediately.</p>
                </div>
                <div class="grid">
                    {% for name in feature_names %}
                    <label>
                        {{ name.replace("_", " ").title() }}
                        <input
                            type="number"
                            name="{{ name }}"
                            value="{{ values.get(name, '') }}"
                            step="any"
                            required
                        >
                    </label>
                    {% endfor %}
                </div>
                <div class="actions">
                    <button type="submit">Predict Diagnosis</button>
                    <a class="button-link" href="/">Reset Sample Values</a>
                </div>
            </form>
            <aside class="side-panel">
                <div class="panel-section">
                    <h2>Output</h2>
                    <p>Submit the form to view the predicted class and probability scores.</p>
                </div>
                <div class="result">
                    {% if error %}
                    <div class="error">{{ error }}</div>
                    {% elif result %}
                    <div class="result-card {{ result.prediction.lower() }}">
                        <div class="result-label">Prediction</div>
                        <div class="result-value">{{ result.prediction }}</div>
                        {% if result.probability_benign is defined %}
                        <div class="probability-row">
                            <span>Benign probability</span>
                            <strong>{{ "%.2f"|format(result.probability_benign * 100) }}%</strong>
                        </div>
                        <div class="probability-row">
                            <span>Malignant probability</span>
                            <strong>{{ "%.2f"|format(result.probability_malignant * 100) }}%</strong>
                        </div>
                        {% endif %}
                    </div>
                    {% else %}
                    <p>No prediction yet.</p>
                    {% endif %}
                </div>
                <div class="panel-section">
                    <h2>Model</h2>
                    <div class="metric"><span>Best model</span><span>SVM (RBF)</span></div>
                    <div class="metric"><span>Test accuracy</span><span>97.37%</span></div>
                    <div class="metric"><span>ROC-AUC</span><span>99.47%</span></div>
                </div>
            </aside>
        </div>
    </main>
</body>
</html>
"""

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en" data-theme="light">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Breast Cancer Detection Dashboard</title>
    <style>
        :root {
            color-scheme: light;
            --page: #edf2f5;
            --panel: #ffffff;
            --ink: #172033;
            --muted: #617086;
            --line: #d6dee8;
            --accent: #0f7c80;
            --accent-dark: #095f63;
            --danger: #b42318;
            --danger-bg: #fff1f0;
            --success: #167044;
            --success-bg: #eefaf3;
            --warning: #7a4d0b;
            --warning-bg: #fff4df;
            --shadow: 0 14px 38px rgba(20, 35, 55, .10);
        }
        html[data-theme="dark"] {
            color-scheme: dark;
            --page: #111820;
            --panel: #18232d;
            --ink: #eff5f7;
            --muted: #aab8c4;
            --line: #2b3a48;
            --accent: #38b6aa;
            --accent-dark: #62d6ca;
            --danger: #ff8d84;
            --danger-bg: #3a1e21;
            --success: #73dfa6;
            --success-bg: #163126;
            --warning: #ffd08a;
            --warning-bg: #33281a;
            --shadow: 0 14px 38px rgba(0, 0, 0, .28);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            background: var(--page);
            color: var(--ink);
        }
        header {
            background: #102938;
            color: white;
            padding: 26px clamp(16px, 4vw, 44px);
        }
        .header-inner {
            max-width: 1240px;
            margin: 0 auto;
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 18px;
            flex-wrap: wrap;
        }
        h1 { margin: 0 0 8px; font-size: clamp(30px, 4vw, 46px); line-height: 1.05; letter-spacing: 0; }
        h2 { margin: 0 0 10px; font-size: 20px; letter-spacing: 0; }
        h3 { margin: 0 0 10px; font-size: 16px; letter-spacing: 0; }
        p { line-height: 1.5; }
        header p { margin: 0; color: #c7d6df; max-width: 760px; }
        .header-tools { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        .link-button, button {
            border: 0;
            border-radius: 6px;
            padding: 10px 13px;
            font-weight: 700;
            font-size: 14px;
            text-decoration: none;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 40px;
        }
        .header-tools a, .theme-toggle {
            color: white;
            background: rgba(255,255,255,.12);
            border: 1px solid rgba(255,255,255,.28);
        }
        main {
            max-width: 1240px;
            margin: 0 auto;
            padding: 22px clamp(16px, 4vw, 44px) 46px;
        }
        .notice {
            background: var(--warning-bg);
            color: var(--warning);
            border: 1px solid color-mix(in srgb, var(--warning) 35%, transparent);
            border-radius: 6px;
            padding: 12px 14px;
            margin-bottom: 18px;
        }
        .dashboard {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 360px;
            gap: 18px;
            align-items: start;
        }
        .panel, form {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: var(--shadow);
        }
        .panel { padding: 18px; margin-bottom: 18px; }
        .form-head { padding: 18px; border-bottom: 1px solid var(--line); }
        .form-head p, .muted { color: var(--muted); margin: 0; }
        .input-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            padding: 18px;
        }
        label { display: grid; gap: 6px; color: var(--muted); font-size: 13px; min-width: 0; }
        input {
            width: 100%;
            border: 1px solid var(--line);
            border-radius: 6px;
            padding: 10px 11px;
            font-size: 15px;
            background: color-mix(in srgb, var(--panel) 92%, var(--page));
            color: var(--ink);
        }
        input:focus { border-color: var(--accent); outline: 3px solid color-mix(in srgb, var(--accent) 22%, transparent); }
        input.invalid { border-color: var(--danger); background: var(--danger-bg); }
        .field-error { color: var(--danger); font-size: 12px; min-height: 15px; }
        .range { font-size: 12px; color: var(--muted); }
        .actions { display: flex; gap: 10px; flex-wrap: wrap; padding: 0 18px 18px; }
        button.primary { background: var(--accent); color: white; }
        .link-button.secondary { background: color-mix(in srgb, var(--accent) 13%, var(--panel)); color: var(--accent-dark); }
        .error {
            background: var(--danger-bg);
            color: var(--danger);
            border: 1px solid color-mix(in srgb, var(--danger) 38%, transparent);
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 12px;
        }
        .result-card {
            border-radius: 8px;
            padding: 18px;
            border: 1px solid var(--line);
            background: color-mix(in srgb, var(--panel) 86%, var(--page));
        }
        .result-card.benign { background: var(--success-bg); border-color: color-mix(in srgb, var(--success) 35%, transparent); }
        .result-card.malignant { background: var(--danger-bg); border-color: color-mix(in srgb, var(--danger) 35%, transparent); }
        .diagnosis-label { color: var(--muted); font-size: 13px; font-weight: 700; text-transform: uppercase; }
        .diagnosis { font-size: 38px; font-weight: 800; margin: 8px 0 12px; }
        .benign .diagnosis { color: var(--success); }
        .malignant .diagnosis { color: var(--danger); }
        .progress { height: 12px; background: color-mix(in srgb, var(--line) 65%, transparent); border-radius: 999px; overflow: hidden; margin: 9px 0 12px; }
        .progress span { display: block; height: 100%; background: var(--accent); border-radius: inherit; }
        .result-row, .metric, .history-row {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            padding: 9px 0;
            border-bottom: 1px solid var(--line);
        }
        .result-row:last-child, .metric:last-child { border-bottom: 0; }
        .bar-row { display: grid; grid-template-columns: minmax(140px, 1fr) 1.5fr; gap: 12px; align-items: center; margin: 10px 0; }
        .bar-track { height: 12px; background: color-mix(in srgb, var(--line) 65%, transparent); border-radius: 999px; overflow: hidden; }
        .bar-fill { height: 100%; background: var(--accent); border-radius: inherit; }
        .table-wrap { overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; min-width: 420px; }
        th, td { padding: 10px 9px; border-bottom: 1px solid var(--line); text-align: left; }
        th { color: var(--muted); font-size: 13px; }
        .matrix { min-width: 360px; text-align: center; }
        .matrix th, .matrix td { text-align: center; }
        .gallery { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
        figure { margin: 0; }
        figure img { width: 100%; border: 1px solid var(--line); border-radius: 8px; background: white; display: block; }
        figcaption { margin-top: 8px; color: var(--muted); font-size: 13px; }
        .info-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
        .stat {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px;
            background: color-mix(in srgb, var(--panel) 88%, var(--page));
        }
        .stat strong { display: block; font-size: 24px; margin-top: 4px; }
        .risk-list { margin: 0; padding-left: 18px; color: var(--muted); line-height: 1.7; }
        @media (max-width: 980px) {
            .dashboard { grid-template-columns: 1fr; }
            .input-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .info-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 640px) {
            .input-grid, .gallery, .info-grid { grid-template-columns: 1fr; }
            .actions, .header-tools { align-items: stretch; }
            .link-button, button { width: 100%; }
            .bar-row { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <header>
        <div class="header-inner">
            <div>
                <h1>Breast Cancer Detection</h1>
                <p>Prediction form, model evidence, testing results, graphs, history, and downloadable reports in one dashboard.</p>
            </div>
            <div class="header-tools">
                <a class="link-button" href="/health">Health</a>
                <a class="link-button" href="/features">Features</a>
                <button class="theme-toggle" type="button" onclick="toggleTheme()">Dark Mode</button>
            </div>
        </div>
    </header>
    <main>
        <div class="notice">Educational demo only. This tool is not medical advice and cannot replace clinician review.</div>
        <div class="dashboard">
            <section>
                <form method="post" action="/" novalidate>
                    <div class="form-head">
                        <h2>Prediction Form</h2>
                        <p>Values must be non-negative and within the observed dataset range.</p>
                    </div>
                    <div class="input-grid">
                        {% for name in feature_names %}
                        <label>
                            {{ name.replace("_", " ").title() }}
                            <input
                                class="{% if field_errors.get(name) %}invalid{% endif %}"
                                type="number"
                                name="{{ name }}"
                                value="{{ values.get(name, '') }}"
                                min="0"
                                step="any"
                                required
                            >
                            <span class="range">{{ feature_ranges[name].min|round(4) }} to {{ feature_ranges[name].max|round(4) }}</span>
                            <span class="field-error">{{ field_errors.get(name, "") }}</span>
                        </label>
                        {% endfor %}
                    </div>
                    <div class="actions">
                        <button class="primary" type="submit">Predict Diagnosis</button>
                        <a class="link-button secondary" href="/">Reset Sample Values</a>
                        {% if result and prediction_id %}
                        <a class="link-button secondary" href="/report/{{ prediction_id }}">Download PDF Report</a>
                        {% endif %}
                    </div>
                </form>

                <div class="panel">
                    <h2>Graphs</h2>
                    <div class="gallery">
                        <figure><img src="{{ assets.class_distribution }}" alt="Class distribution"><figcaption>Class distribution</figcaption></figure>
                        <figure><img src="{{ assets.correlation_heatmap }}" alt="Correlation heatmap"><figcaption>Correlation heatmap</figcaption></figure>
                        <figure><img src="{{ assets.top_feature_boxplots }}" alt="Feature distribution"><figcaption>Feature distribution</figcaption></figure>
                        <figure><img src="{{ assets.roc_curve }}" alt="ROC curve"><figcaption>ROC curve</figcaption></figure>
                    </div>
                </div>

                <div class="panel">
                    <h2>Model Comparison</h2>
                    <div class="table-wrap">
                        <table>
                            <thead><tr><th>Model</th><th>Cross-Validation Accuracy</th></tr></thead>
                            <tbody>
                                {% for model_name, model_result in model_comparison.items() %}
                                <tr>
                                    <td>{{ model_name }}</td>
                                    <td>{{ "%.2f"|format(model_result.mean_cv_accuracy * 100) }}%</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="panel">
                    <h2>Confusion Matrix</h2>
                    <div class="table-wrap">
                        <table class="matrix">
                            <thead>
                                <tr><th></th><th colspan="2">Predicted</th></tr>
                                <tr><th>Actual</th><th>Benign</th><th>Malignant</th></tr>
                            </thead>
                            <tbody>
                                <tr><th>Benign</th><td>{{ confusion_matrix[0][0] }}</td><td>{{ confusion_matrix[0][1] }}</td></tr>
                                <tr><th>Malignant</th><td>{{ confusion_matrix[1][0] }}</td><td>{{ confusion_matrix[1][1] }}</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>

            <aside>
                <div class="panel">
                    <h2>Diagnosis</h2>
                    {% if error %}<div class="error">{{ error }}</div>{% endif %}
                    {% if result %}
                    <div class="result-card {{ result.prediction.lower() }}">
                        <div class="diagnosis-label">Diagnosis</div>
                        <div class="diagnosis">{{ result.prediction }}</div>
                        <strong>Confidence: {{ "%.2f"|format(result.confidence * 100) }}%</strong>
                        <div class="progress"><span style="width: {{ result.confidence * 100 }}%"></span></div>
                        <div class="result-row"><span>Benign</span><strong>{{ "%.2f"|format(result.probability_benign * 100) }}%</strong></div>
                        <div class="result-row"><span>Malignant</span><strong>{{ "%.2f"|format(result.probability_malignant * 100) }}%</strong></div>
                    </div>
                    {% else %}
                    <p class="muted">Submit the form to see the prediction and confidence score.</p>
                    {% endif %}
                </div>

                <div class="panel">
                    <h2>Top Features</h2>
                    {% if result %}
                        {% for item in result.top_features %}
                        <div class="bar-row">
                            <span>{{ item.name.replace("_", " ").title() }}</span>
                            <div class="bar-track"><div class="bar-fill" style="width: {{ item.percent }}%"></div></div>
                        </div>
                        {% endfor %}
                    {% else %}
                        <p class="muted">Top contributing features appear after prediction.</p>
                    {% endif %}
                </div>

                <div class="panel">
                    <h2>Prediction History</h2>
                    {% if history %}
                        {% for row in history %}
                        <div class="history-row">
                            <span>{{ row.created_at[5:10] }} {{ row.created_at[11:16] }}</span>
                            <strong>{{ row.prediction }}</strong>
                            <span>{{ "%.1f"|format(row.confidence * 100) }}%</span>
                        </div>
                        {% endfor %}
                    {% else %}
                    <p class="muted">No saved predictions yet.</p>
                    {% endif %}
                </div>

                <div class="panel">
                    <h2>Dataset</h2>
                    <div class="info-grid">
                        <div class="stat"><span>Samples</span><strong>{{ dataset_info.samples }}</strong></div>
                        <div class="stat"><span>Features</span><strong>{{ dataset_info.features }}</strong></div>
                        <div class="stat"><span>Benign</span><strong>{{ dataset_info.benign }}</strong></div>
                        <div class="stat"><span>Malignant</span><strong>{{ dataset_info.malignant }}</strong></div>
                    </div>
                    <p class="muted" style="margin-top: 12px;">Train/Test: {{ dataset_info.train_percent }}/{{ dataset_info.test_percent }}</p>
                </div>

                <div class="panel">
                    <h2>About Model</h2>
                    <div class="metric"><span>Model</span><strong>Support Vector Machine</strong></div>
                    <div class="metric"><span>Kernel</span><strong>RBF</strong></div>
                    <div class="metric"><span>Scaling</span><strong>StandardScaler</strong></div>
                    <div class="metric"><span>Test Accuracy</span><strong>{{ "%.2f"|format(test_metrics.accuracy * 100) }}%</strong></div>
                    <div class="metric"><span>Precision</span><strong>{{ "%.2f"|format(test_metrics.precision * 100) }}%</strong></div>
                    <div class="metric"><span>Recall</span><strong>{{ "%.2f"|format(test_metrics.recall * 100) }}%</strong></div>
                    <div class="metric"><span>ROC-AUC</span><strong>{{ "%.2f"|format(test_metrics.roc_auc * 100) }}%</strong></div>
                </div>

                <div class="panel">
                    <h2>Health Information</h2>
                    <h3>Risk Factors</h3>
                    <ul class="risk-list">
                        <li>Age</li>
                        <li>Family history</li>
                        <li>Smoking</li>
                        <li>Obesity</li>
                    </ul>
                </div>
            </aside>
        </div>
    </main>
    <script>
        const savedTheme = localStorage.getItem("theme");
        if (savedTheme) document.documentElement.dataset.theme = savedTheme;
        function toggleTheme() {
            const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
            document.documentElement.dataset.theme = next;
            localStorage.setItem("theme", next);
        }
    </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE_TEMPLATE, **template_context())


@app.route("/", methods=["POST"])
def predict_from_form():
    values = {f: request.form.get(f, "") for f in feature_names}
    result, error = run_prediction(values)
    prediction_id = save_prediction(result) if result else None
    return render_template_string(
        PAGE_TEMPLATE,
        **template_context(
            values=values,
            result=result,
            error=error["error"] if error else None,
            field_errors=error.get("field_errors", {}) if error else {},
            prediction_id=prediction_id,
        ),
    )


@app.route("/outputs/<path:filename>")
def output_file(filename):
    allowed = {
        "class_distribution.png",
        "correlation_heatmap.png",
        "top_feature_boxplots.png",
        "roc_curve.png",
        "feature_importance.png",
    }
    if filename not in allowed:
        return jsonify({"error": "File not found"}), 404
    return send_file(os.path.join(OUTPUTS_DIR, filename))


@app.route("/report/<int:prediction_id>")
def download_report(prediction_id):
    record = get_prediction_record(prediction_id)
    if record is None:
        return jsonify({"error": "Report not found"}), 404
    return send_file(
        build_pdf_report(record),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"prediction_report_{prediction_id}.pdf",
    )


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

    result, error = run_prediction(features_dict)
    if error:
        return jsonify(error), 400

    result["prediction_id"] = save_prediction(result)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
