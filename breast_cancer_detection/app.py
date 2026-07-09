from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


DATA_PATH = Path("breast-cancer.csv")
MODEL_PATH = Path("model.joblib")
FEATURES_PATH = Path("features.joblib")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        st.error("Model files are missing. Run: python train_model.py")
        st.stop()

    return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)


@st.cache_data
def load_reference_data():
    data = pd.read_csv(DATA_PATH)
    return data.drop(columns=["id", "diagnosis"], errors="ignore")


def prediction_badge(label: str, probability: float):
    if label == "Malignant":
        st.error(f"Prediction: {label}")
    else:
        st.success(f"Prediction: {label}")

    st.metric("Malignant probability", f"{probability * 100:.2f}%")


def main():
    st.set_page_config(page_title="Breast Cancer Detection", layout="wide")
    st.title("Breast Cancer Detection")

    model, features = load_model()
    reference = load_reference_data()

    st.sidebar.header("Patient Measurements")
    mode = st.sidebar.radio("Input method", ["Manual entry", "Upload CSV"])

    if mode == "Manual entry":
        values = {}
        columns = st.sidebar.columns(2)

        for index, feature in enumerate(features):
            series = reference[feature]
            values[feature] = columns[index % 2].number_input(
                feature,
                min_value=float(series.min()),
                max_value=float(series.max()),
                value=float(series.median()),
                step=float((series.max() - series.min()) / 100),
            )

        input_data = pd.DataFrame([values], columns=features)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        label = "Malignant" if prediction == 1 else "Benign"

        left, right = st.columns([1, 2])
        with left:
            prediction_badge(label, probability)
        with right:
            st.subheader("Entered Values")
            st.dataframe(input_data, use_container_width=True)

    else:
        uploaded_file = st.file_uploader("Upload a CSV file with the same feature columns", type=["csv"])
        if uploaded_file is None:
            st.info("Upload a CSV to generate predictions.")
            return

        uploaded = pd.read_csv(uploaded_file)
        input_data = uploaded.reindex(columns=features)

        if input_data.isna().any().any():
            missing = [feature for feature in features if feature not in uploaded.columns]
            st.error(f"Missing feature columns: {missing}")
            return

        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]

        results = uploaded.copy()
        results["prediction"] = ["Malignant" if value == 1 else "Benign" for value in predictions]
        results["malignant_probability"] = probabilities

        st.subheader("Prediction Results")
        st.dataframe(results, use_container_width=True)
        st.download_button(
            "Download predictions",
            results.to_csv(index=False).encode("utf-8"),
            file_name="breast_cancer_predictions.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
