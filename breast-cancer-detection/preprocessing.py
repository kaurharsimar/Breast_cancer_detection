"""
preprocessing.py
-----------------
Loads the raw Wisconsin Breast Cancer dataset, cleans it, encodes the
target label, and splits it into train/test sets. Also fits and saves
a StandardScaler so the exact same scaling can be reused at inference
time (in the API).

Run directly to sanity-check the pipeline:
    python preprocessing.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

DATA_PATH = os.path.join("data", "breast-cancer.csv")
MODELS_DIR = "models"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop identifier column and any stray "Unnamed" columns some
    # copies of this dataset ship with (an artifact of a trailing comma).
    drop_cols = [c for c in df.columns if c == "id" or c.lower().startswith("unnamed")]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    # Malignant = 1 (positive / the class we care most about catching),
    # Benign = 0
    df = df.copy()
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, save_scaler: bool = True):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_scaler:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    return X_train_scaled, X_test_scaled, scaler


def run_pipeline():
    df = load_data()
    df = encode_target(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Persist the feature column order — the API needs this to build
    # a correctly-ordered feature vector from JSON input.
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(list(X_train.columns), os.path.join(MODELS_DIR, "feature_names.pkl"))

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X_train.columns)


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Class balance:\n{df['diagnosis'].value_counts()}")

    X_train, X_test, y_train, y_test, scaler, feature_names = run_pipeline()
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")
    print("Preprocessing complete. Scaler and feature names saved to models/")
