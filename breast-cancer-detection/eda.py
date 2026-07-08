"""
eda.py
-------
Generates exploratory data analysis plots used in the project report /
presentation: class balance, feature correlations, and top discriminative
features. Saves PNGs to outputs/.

Run:
    python eda.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from preprocessing import load_data, encode_target

OUTPUTS_DIR = "outputs"


def plot_class_balance(df):
    plt.figure(figsize=(5, 4))
    counts = df["diagnosis"].value_counts()
    labels = ["Benign", "Malignant"]
    colors = ["#4CAF50", "#E53935"]
    plt.bar(labels, [counts[0], counts[1]], color=colors)
    plt.title("Class Distribution")
    plt.ylabel("Number of Patients")
    for i, v in enumerate([counts[0], counts[1]]):
        plt.text(i, v + 3, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "class_distribution.png"), dpi=150)
    plt.close()


def plot_correlation_heatmap(df):
    plt.figure(figsize=(14, 12))
    corr = df.drop(columns=["diagnosis"]).corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.7})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()


def plot_top_feature_boxplots(df):
    top_features = ["radius_mean", "perimeter_mean", "area_mean", "concavity_mean", "concave points_mean"]
    fig, axes = plt.subplots(1, len(top_features), figsize=(20, 4))
    for ax, feat in zip(axes, top_features):
        sns.boxplot(x="diagnosis", y=feat, data=df, ax=ax, palette=["#4CAF50", "#E53935"])
        ax.set_xticklabels(["Benign", "Malignant"])
        ax.set_title(feat)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "top_feature_boxplots.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df = load_data()
    df = encode_target(df)

    plot_class_balance(df)
    plot_correlation_heatmap(df)
    plot_top_feature_boxplots(df)

    print(f"EDA plots saved to {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
