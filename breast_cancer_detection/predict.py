import argparse
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("model.joblib")
FEATURES_PATH = Path("features.joblib")


def load_artifacts():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError("Run 'python train_model.py' before predicting.")

    return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)


def predict_from_csv(csv_path: Path) -> pd.DataFrame:
    model, features = load_artifacts()
    data = pd.read_csv(csv_path)
    input_data = data.reindex(columns=features)

    if input_data.isna().any().any():
        missing = [column for column in features if column not in data.columns]
        raise ValueError(f"Input CSV is missing feature columns: {missing}")

    predicted_class = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]

    result = data.copy()
    result["prediction"] = ["Malignant" if value == 1 else "Benign" for value in predicted_class]
    result["malignant_probability"] = probabilities
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict breast cancer diagnosis from a CSV file.")
    parser.add_argument("csv", type=Path, help="Path to a CSV containing the model feature columns.")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"), help="Where to save predictions.")
    args = parser.parse_args()

    result = predict_from_csv(args.csv)
    result.to_csv(args.output, index=False)
    print(result[["prediction", "malignant_probability"]])
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
