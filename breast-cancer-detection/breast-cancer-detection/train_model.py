"""
train_model.py
----------------
Trains and compares several classifiers on the breast cancer dataset,
picks the best performer using 5-fold cross-validation on the training
set, evaluates it on the held-out test set, and saves the winning
model to models/best_model.pkl.

Run:
    python train_model.py
"""

import json
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from preprocessing import run_pipeline

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"


def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    }


def compare_models(X_train, y_train):
    results = {}
    for name, model in get_candidate_models().items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        results[name] = {"mean_cv_accuracy": scores.mean(), "std_cv_accuracy": scores.std()}
        print(f"{name:22s}  CV accuracy = {scores.mean():.4f}  (+/- {scores.std():.4f})")
    return results


def tune_best_model(best_name, X_train, y_train):
    """Light hyperparameter tuning for whichever model wins the CV comparison."""
    param_grids = {
        "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100]},
        "SVM (RBF)": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.001]},
        "Random Forest": {"n_estimators": [200, 300, 500], "max_depth": [None, 5, 10, 15]},
        "Gradient Boosting": {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1]},
        "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7, 9, 11]},
    }
    base_model = get_candidate_models()[best_name]
    grid = GridSearchCV(base_model, param_grids[best_name], cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"\nBest params for {best_name}: {grid.best_params_}")
    print(f"Best CV accuracy after tuning: {grid.best_score_:.4f}")
    return grid.best_estimator_


def evaluate_on_test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Test Set Performance ===")
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted) [0=Benign, 1=Malignant]")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    return metrics, cm.tolist()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, scaler, feature_names = run_pipeline()

    print("=== Cross-Validation Comparison (5-fold, on training data) ===")
    cv_results = compare_models(X_train, y_train)

    best_name = max(cv_results, key=lambda k: cv_results[k]["mean_cv_accuracy"])
    print(f"\nBest model by CV accuracy: {best_name}")

    best_model = tune_best_model(best_name, X_train, y_train)
    best_model.fit(X_train, y_train)

    test_metrics, cm = evaluate_on_test(best_model, X_test, y_test)

    # Save the trained model
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))

    # Save a summary report
    report = {
        "best_model": best_name,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "n_features": len(feature_names),
        "feature_names": feature_names,
    }
    with open(os.path.join(OUTPUTS_DIR, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved trained model to {MODELS_DIR}/best_model.pkl")
    print(f"Saved training report to {OUTPUTS_DIR}/training_report.json")


if __name__ == "__main__":
    main()
