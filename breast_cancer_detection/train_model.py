from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("breast-cancer.csv")
MODEL_PATH = Path("model.joblib")
FEATURES_PATH = Path("features.joblib")
METRICS_PATH = Path("metrics.txt")


def load_dataset(path: Path = DATA_PATH):
    data = pd.read_csv(path)
    data = data.drop(columns=["id"], errors="ignore")

    if "diagnosis" not in data.columns:
        raise ValueError("Dataset must contain a 'diagnosis' column.")

    x = data.drop(columns=["diagnosis"])
    y = data["diagnosis"].map({"B": 0, "M": 1})

    if y.isna().any():
        raise ValueError("Diagnosis column must contain only 'B' and 'M' values.")

    return x, y


def train():
    x, y = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=["Benign", "Malignant"],
    )
    matrix = confusion_matrix(y_test, predictions)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(x.columns), FEATURES_PATH)

    metrics = (
        f"Accuracy: {accuracy:.4f}\n\n"
        "Classification Report:\n"
        f"{report}\n"
        "Confusion Matrix:\n"
        f"{matrix}\n"
    )
    METRICS_PATH.write_text(metrics, encoding="utf-8")

    print(metrics)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved feature list to {FEATURES_PATH}")


if __name__ == "__main__":
    train()
