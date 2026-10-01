"""Lab 3, Task 1: frequency and impact encoding (student starter)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "lab3" / "credit_approval.csv"


def frequency_encode(x: np.ndarray) -> np.ndarray:
    """Replace each category by its proportion in the supplied 1D array."""
    # For example, if "sunny" occurs 5 times in 14 rows, encode it as 5/14.
    raise NotImplementedError("Implement frequency_encode")


def impact_encode(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Replace each category by the proportion of class 1 within it.

    This lab uses binary labels 0 and 1, so the mean of y within a category
    is also its proportion of class 1.
    """
    raise NotImplementedError("Implement impact_encode")


def check_functions():
    """Print the encodings for the 14-row example on lecture slide 53."""
    outlook = np.array([
        "sunny", "sunny", "overcast", "rain", "rain", "rain",
        "overcast", "sunny", "sunny", "rain", "sunny", "overcast",
        "overcast", "rain",
    ])
    play_golf = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1])

    frequency = frequency_encode(outlook)
    impact = impact_encode(outlook, play_golf)
    print("Outlook    y  Frequency  Impact")
    for category, label, freq, value in zip(outlook, play_golf, frequency, impact):
        print(f"{category:9s}  {label}  {freq:9.3f}  {value:6.3f}")


def is_numeric_col(col: np.ndarray) -> bool:
    """Return whether every value in a column can be converted to float."""
    try:
        np.asarray(col, dtype=float)
        return True
    except (TypeError, ValueError):
        return False


class FrequencyEncoderTransformer(TransformerMixin, BaseEstimator):
    """Encode string columns using frequencies learned during fit.

    Keep numeric columns unchanged. For a category not seen during fit, use
    0.0. The output of transform should be a 2D float NumPy array.
    """

    def fit(self, X, y=None):
        # Store one mapping for each categorical column. Do not learn anything
        # from the test data in transform.
        raise NotImplementedError("Implement FrequencyEncoderTransformer.fit")

    def transform(self, X):
        raise NotImplementedError("Implement FrequencyEncoderTransformer.transform")


class ImpactEncoderTransformer(TransformerMixin, BaseEstimator):
    """Encode string columns using class-1 rates learned during fit.

    Keep numeric columns unchanged. For an unseen category, use the overall
    class-1 rate from fit. The output should be a 2D float NumPy array.
    """

    def fit(self, X, y):
        raise NotImplementedError("Implement ImpactEncoderTransformer.fit")

    def transform(self, X):
        raise NotImplementedError("Implement ImpactEncoderTransformer.transform")


def load_credit_approval_data(path=DATA_PATH):
    """Load credit approval as mixed-type X and binary y.

    The original file contains incomplete rows. Drop them here so Task 1 can
    concentrate on encoding; missing-value imputation was covered in Lab 1.
    """
    data = pd.read_csv(path).dropna()
    X = data.drop(columns="target").to_numpy(dtype=object)
    y = data["target"].to_numpy()
    return X, y


if __name__ == "__main__":
    X, y = load_credit_approval_data()
    print(f"Credit approval: {X.shape[0]} complete rows, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")
    print("Implement the helpers, then call check_functions() to inspect them.")
