# 🩺 Breast Cancer Detection using Machine Learning

## 📌 Overview

This project is a Machine Learning-based Breast Cancer Detection System that classifies breast tumors as **Benign** or **Malignant** using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset. Multiple machine learning algorithms were trained and compared to identify the best-performing model. The selected model is deployed using a **Flask REST API**, allowing predictions through HTTP requests.

> **Disclaimer:** This project is developed for educational and research purposes only. It should not be used for real medical diagnosis.

---

# 🚀 Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature scaling using StandardScaler
- Comparison of multiple Machine Learning algorithms
- Hyperparameter tuning using GridSearchCV
- Model evaluation using various performance metrics
- Model persistence using Joblib
- REST API using Flask
- Easy deployment and integration

---

# 📂 Dataset

**Dataset Name:** Wisconsin Diagnostic Breast Cancer (WDBC)

- Total Samples: **569**
- Benign Cases: **357**
- Malignant Cases: **212**
- Number of Features: **30**
- Missing Values: **None**

The dataset contains diagnostic measurements extracted from digitized images of Fine Needle Aspirate (FNA) of breast masses.

### Features include:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave Points
- Symmetry
- Fractal Dimension

Each measurement includes:

- Mean
- Standard Error (SE)
- Worst Value

---

# 📁 Project Structure

```
breast-cancer-detection/
│
├── api/
│   └── app.py
│
├── data/
│   └── breast-cancer.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── outputs/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── top_feature_boxplots.png
│   └── training_report.json
│
├── preprocessing.py
├── train_model.py
├── eda.py
├── sample_request.json
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🛠 Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Flask
- Joblib

---

# 🔄 Machine Learning Workflow

1. Load Dataset
2. Data Cleaning
3. Encode Target Labels
4. Train-Test Split
5. Feature Scaling
6. Train Multiple Models
7. Hyperparameter Tuning
8. Evaluate Model
9. Save Best Model
10. Deploy with Flask API

---

# 🤖 Models Compared

| Model | Cross Validation Accuracy |
|--------|--------------------------|
| Logistic Regression | 97.14% |
| Support Vector Machine (RBF) | **97.58%** |
| K-Nearest Neighbors | 96.92% |
| Random Forest | 95.82% |
| Gradient Boosting | 95.38% |

The **Support Vector Machine (RBF Kernel)** achieved the highest cross-validation accuracy and was selected as the final model.

---

# 📈 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **97.37%** |
| Precision | **100.00%** |
| Recall | **92.86%** |
| F1 Score | **96.30%** |
| ROC-AUC | **99.47%** |

### Confusion Matrix

| Actual | Predicted Benign | Predicted Malignant |
|---------|------------------|---------------------|
| Benign | 72 | 0 |
| Malignant | 3 | 39 |

---

# 📊 Exploratory Data Analysis

The project includes:

- Class Distribution
- Feature Correlation Heatmap
- Important Feature Boxplots

These visualizations help understand the data before training the machine learning models.

---

# ⚙ Installation

## Clone the Repository

```bash
git clone https://github.com/yourusername/breast-cancer-detection.git
```

## Navigate to the Project Folder

```bash
cd breast-cancer-detection
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Project

### Step 1: Preprocess the Dataset

```bash
python preprocessing.py
```

### Step 2: Generate EDA Plots

```bash
python eda.py
```

### Step 3: Train the Machine Learning Model

```bash
python train_model.py
```

### Step 4: Start the Flask API

```bash
python api/app.py
```

The server will run on

```
http://127.0.0.1:5000
```

---

# 🌐 API Endpoints

## Health Check

```
GET /health
```

Example

```
http://127.0.0.1:5000/health
```

Response

```json
{
    "status": "ok",
    "model_loaded": true
}
```

---

## List Required Features

```
GET /features
```

Returns all 30 features expected by the model.

---

## Predict Breast Cancer

```
POST /predict
```

Content-Type

```
application/json
```

Example Request

```json
{
  "features": {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    "area_mean": 1001,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095,
    "texture_se": 0.9053,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38,
    "texture_worst": 17.33,
    "perimeter_worst": 184.6,
    "area_worst": 2019,
    "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.1189
  }
}
```

Example Response

```json
{
    "prediction": "Malignant",
    "prediction_code": 1,
    "probability_benign": 0.0189,
    "probability_malignant": 0.9811
}
```

---

# 📷 Project Output

The project generates:

- Class Distribution Plot
- Correlation Heatmap
- Feature Boxplots
- Training Report
- Flask Prediction API

---

# 📌 Future Improvements

- Build a responsive web interface
- Deploy using Render or Railway
- Docker support
- SHAP Explainable AI
- User authentication
- Prediction history
- Batch prediction using CSV upload

---

# 👨‍💻 Author

**Harsimar Kaur**

**B.Tech - Artificial Intelligence & Machine Learning**

---

# ⭐ If you found this project helpful, don't forget to give it a Star on GitHub!
