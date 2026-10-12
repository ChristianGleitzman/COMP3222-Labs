"""Lab 3, Task 2: compare regression models (student starter)."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

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

    # Fit LinearRegression on X_train, y_train.
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Print mean_squared_error for training and test predictions.
    train_predict = lr.predict(X_train)
    test_predict = lr.predict(X_test)
    print("========= Linear Regression ==========")
    print(f"MSE on train (lr): {mean_squared_error(y_train, train_predict)}")
    print(f"MSE on test (lr): {mean_squared_error(y_test, test_predict)}")


    print("========= Tree Regression ==========")
    # Repeat for DecisionTreeRegressor(max_depth=3, random_state=42).
    for i in range(1,10):
        tr = DecisionTreeRegressor(max_depth=i, random_state=42)
        tr.fit(X_train, y_train)

        # Print mean_squared_error for training and test predictions.
        train_predict = tr.predict(X_train)
        test_predict = tr.predict(X_test)
        print(f"MSE on train (tree depth {i}): {mean_squared_error(y_train, train_predict)}")
        print(f"MSE on test (tree depth {i}): {mean_squared_error(y_test, test_predict)}")
        print()

    # Tree depth 5 seems to be the sweet spot
    # Tree begins to overfit on training data, performing worse on unseen data

    print("========= Forest Regression ==========")

    rfr = RandomForestRegressor(random_state=42)
    rfr.fit(X_train, y_train)

    # Print mean_squared_error for training and test predictions.
    train_predict = rfr.predict(X_train)
    test_predict = rfr.predict(X_test)
    print(f"MSE on forest: {mean_squared_error(y_train, train_predict)}")
    print(f"MSE on test: {mean_squared_error(y_test, test_predict)}")
    print()
