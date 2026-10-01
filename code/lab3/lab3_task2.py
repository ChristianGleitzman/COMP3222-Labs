"""Lab 3, Task 2: compare regression models (student starter)."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "lab3" / "winequality-red.csv"


def load_wine_quality_data(path=DATA_PATH):
    """Return wine measurements X and quality scores y as NumPy arrays."""
    data = pd.read_csv(path, sep=";")
    X = data.drop(columns="quality").to_numpy()
    y = data["quality"].to_numpy()
    return X, y


if __name__ == "__main__":
    X, y = load_wine_quality_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42
    )
    print(f"Wine quality: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Training: {len(y_train)} rows; test: {len(y_test)} rows")

    # TODO: Fit LinearRegression on X_train, y_train.
    # TODO: Print mean_squared_error for training and test predictions.
    # TODO: Repeat for DecisionTreeRegressor(max_depth=3, random_state=42).
    # TODO: Try several tree depths, then RandomForestRegressor(random_state=42).
