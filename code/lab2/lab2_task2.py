"""Lab 2, Task 2 starter: fit and visualise a decision tree on PlayGolf.

Run from the repository root with: python code/lab2/lab2_task2.py
Complete the steps below using Lab Sheet 2. The data-loading step is provided.
"""

from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "lab2" / "playgolf.csv"


def load_playgolf():
    """Return the four attributes as a DataFrame and the labels as an array."""
    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns="PlayGolf")
    y = data["PlayGolf"].to_numpy()
    return X, y


if __name__ == "__main__":
    X, y = load_playgolf()
    print("Attributes:", list(X.columns))
    print("Raw X shape:", X.shape, "y shape:", y.shape)
    print("Outlook values:", X["Outlook"].unique())

    # A. Try fitting DecisionTreeClassifier on X as it is. What error do you get?
    # Then one-hot encode Outlook, keeping Temp, Humidity, and Windy numeric.
    # ColumnTransformer and OneHotEncoder can do this in a Pipeline.
    # Check that the transformed X has shape (14, 6).

    # B. Fit a DecisionTreeClassifier to the numeric features and y.
    # Use sklearn.tree.plot_tree with feature_names and class_names.
    # If you used a Pipeline, plot its fitted tree step, not the Pipeline itself.
    # How does this tree compare with the one from the lecture?

    # Try ExtraTreeClassifier as well. Compare the two trees and their
    # training predictions. What would you need to compare them fairly on
    # new data?
