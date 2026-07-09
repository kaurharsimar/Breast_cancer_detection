# Breast Cancer Detection — AI/ML Capstone Project

A machine learning system that classifies breast tumors as **Malignant** or
**Benign** from diagnostic measurements, served through a REST API.

## Dataset

**Wisconsin Diagnostic Breast Cancer (WDBC)** dataset — 569 patient records,
30 numeric features computed from digitized images of a fine needle
aspirate (FNA) of a breast mass (radius, texture, perimeter, area,
smoothness, compactness, concavity, concave points, symmetry, fractal
dimension — each reported as mean, standard error, and "worst" value).

- Target: `diagnosis` — `M` (malignant) or `B` (benign)
- Class balance: 357 benign, 212 malignant
- No missing values

## Project Structure

```
breast-cancer-detection/
├── data/
│   └── breast-cancer.csv       # raw dataset
├── models/
│   ├── best_model.pkl          # trained classifier
│   ├── scaler.pkl              # fitted StandardScaler
│   └── feature_names.pkl       # ordered feature list expected by the model
├── outputs/
│   ├── training_report.json    # CV results, test metrics, confusion matrix
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   └── top_feature_boxplots.png
├── api/
│   └── app.py                  # Flask REST API
├── preprocessing.py            # data loading, cleaning, scaling
├── train_model.py              # model comparison, tuning, training
├── eda.py                      # exploratory data analysis plots
├── sample_request.json         # example API request payload
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess & sanity-check the data
```bash
python preprocessing.py
```

### 2. Run exploratory data analysis
```bash
python eda.py
```
Generates plots in `outputs/`: class balance, correlation heatmap, and
boxplots of the most discriminative features by diagnosis.

### 3. Train the model
```bash
python train_model.py
```
This trains and 5-fold cross-validates five candidate models (Logistic
Regression, SVM, Random Forest, Gradient Boosting, KNN), tunes the
best one with `GridSearchCV`, evaluates it on a held-out test set, and
saves the winning model to `models/best_model.pkl`.

### 4. Open the HTML prediction page / serve the API
```bash
python run_web.py
```
Runs on `http://localhost:5000`.

Open `http://localhost:5000` in your browser to use the HTML prediction
form. The form is prefilled with the sample request values, so you can
click **Predict Diagnosis** immediately to see the model output on the page.

The website also includes input validation, confidence scoring, top feature
bars, prediction history in SQLite, PDF report downloads, model comparison,
confusion matrix, ROC curve, dataset information, project graphs, dark mode,
and an educational health note.

**Endpoints:**

| Method | Route       | Description                                  |
|--------|-------------|-----------------------------------------------|
| GET    | `/health`   | Liveness check                                |
| GET    | `/features` | Lists the 30 expected input features, in order|
| POST   | `/predict`  | Returns a diagnosis prediction                |

**Example request:**
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d @sample_request.json
```

**Example response:**
```json
{
  "prediction": "Malignant",
  "prediction_code": 1,
  "probability_benign": 0.0189,
  "probability_malignant": 0.9811
}
```

## Model Results

Five models were compared with 5-fold cross-validation on the training
set; the best (SVM with RBF kernel) was then hyperparameter-tuned.

| Model               | CV Accuracy |
|---------------------|-------------|
| **SVM (RBF)**       | **0.9758**  |
| Logistic Regression | 0.9714      |
| KNN                 | 0.9692      |
| Random Forest       | 0.9582      |
| Gradient Boosting   | 0.9538      |

**Final test set performance (SVM, tuned, C=1, gamma='scale'):**

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9737 |
| Precision | 1.0000 |
| Recall    | 0.9286 |
| F1-score  | 0.9630 |
| ROC-AUC   | 0.9947 |

Confusion matrix (test set, 114 samples):

|                  | Predicted Benign | Predicted Malignant |
|------------------|------------------|----------------------|
| **Actual Benign**    | 72 | 0  |
| **Actual Malignant** | 3  | 39 |

Precision of 1.0 means every tumor the model flagged malignant was
actually malignant (zero false alarms on this test split); the 3 missed
malignant cases (false negatives) are the main thing to keep watching if
you extend this project, since in a real screening context recall on
the malignant class matters most.

## Possible Extensions (for GitHub / deployment deliverables)

- Add a simple HTML/React frontend that posts to `/predict`
- Containerize with Docker and deploy to Render / Railway / AWS Elastic Beanstalk
- Add SHAP-based explainability so predictions show which features drove them
- Add input validation ranges based on training data min/max
- Log predictions to a database for monitoring/drift detection

## Notes

This project is for educational purposes and is **not a diagnostic tool**.
Any real clinical application would require regulatory approval, much
larger and more diverse validation data, and clinician oversight.
