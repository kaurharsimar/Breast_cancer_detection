# Breast Cancer Detection

This project trains a machine learning model on `breast-cancer.csv` to classify tumors as:

- `Benign`
- `Malignant`

The dataset uses the `diagnosis` column as the target label, where `B` means benign and `M` means malignant.

## Project Files

- `breast-cancer.csv` - dataset
- `train_model.py` - trains and saves the model
- `app.py` - Streamlit web app for predictions
- `predict.py` - command-line CSV prediction script
- `requirements.txt` - Python dependencies
- `metrics.txt` - generated after training
- `model.joblib` and `features.joblib` - generated model files

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train_model.py
```

This creates:

- `model.joblib`
- `features.joblib`
- `metrics.txt`

## Run the Web App

```bash
streamlit run app.py
```

The app supports manual prediction and batch CSV prediction.

## Predict from a CSV

```bash
python predict.py breast-cancer.csv --output predictions.csv
```

The output file includes:

- `prediction`
- `malignant_probability`

## Model Used

The project uses a `RandomForestClassifier` inside a scikit-learn pipeline. The input features are scaled with `StandardScaler`, and the model is trained with a stratified train-test split.
