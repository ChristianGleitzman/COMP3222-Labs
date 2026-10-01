"""Lab 1, Task 2: encoding, imputation, scaling, and a pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "lab1"


def one_hot_categorical():
    """Return numeric features and labels from the mixed-type CSV."""
    df = pd.read_csv(
        DATA_DIR / "one_hot.csv",
        header=None,
        names=["x1", "x2", "x3", "rank", "colour", "y"],
    )
    continuous = df[["x1", "x2", "x3"]].to_numpy(dtype=float)
    rank = df["rank"].to_numpy(dtype=float)
    colour = df["colour"].to_numpy().reshape(-1, 1)  # OneHotEncoder needs 2D input.

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    colour_columns = encoder.fit_transform(colour)
    X = np.column_stack([continuous, rank, colour_columns])
    y = df["y"].to_numpy(dtype=int)
    return X, y


def impute_missing(X):
    """Replace missing values with each feature's median."""
    return SimpleImputer(strategy="median").fit_transform(X)


def standardise(X):
    """Give each feature mean zero and standard deviation one."""
    return StandardScaler().fit_transform(X)


def make_pipeline():
    """Apply the same preparation to training data and future test data."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", KNeighborsClassifier()),
    ])


def fit_evaluate(X, y):
    """Split raw data first so the imputer and scaler see only training rows."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    pipe = make_pipeline()
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    return accuracy_score(y_test, predictions)


if __name__ == "__main__":
    # A: Colour becomes one column per category; every feature is now numeric.
    X_categorical, y_categorical = one_hot_categorical()
    print("One-hot feature matrix:", X_categorical.shape)
    KNeighborsClassifier().fit(X_categorical, y_categorical)
    print("Classifier fitted the numeric features successfully")

    # B and C: Inspect imputation and standardisation on the full dataset.
    # This is a demonstration, not the data used in the held-out evaluation.
    df = pd.read_csv(DATA_DIR / "missing.csv", header=None)
    X_raw = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy(dtype=int)
    X_imputed = impute_missing(X_raw)
    X_scaled = standardise(X_imputed)
    print("Missing values before/after:", np.isnan(X_raw).sum(), np.isnan(X_imputed).sum())
    print("Scaled feature means:", X_scaled.mean(axis=0).round(3))

    # D and E: Start again from raw data. The pipeline learns imputation and
    # scaling from X_train only, then applies them to X_test.
    print(f"Held-out pipeline accuracy: {fit_evaluate(X_raw, y):.3f}")
