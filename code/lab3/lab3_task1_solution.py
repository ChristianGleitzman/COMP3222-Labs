"""Lab 3, Task 1 solution: frequency and impact encoding."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from lab3_task1 import is_numeric_col, load_credit_approval_data


def frequency_encode(x: np.ndarray) -> np.ndarray:
    """Replace each category by its proportion in the supplied 1D array."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return counts[inverse] / len(x)


def impact_encode(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Replace each category by the proportion of class 1 within it."""
    _, inverse = np.unique(x, return_inverse=True)
    class_1_rate = np.bincount(inverse, weights=y) / np.bincount(inverse)
    return class_1_rate[inverse]


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


class FrequencyEncoderTransformer(TransformerMixin, BaseEstimator):
    """Encode string columns using frequencies learned during fit."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=object)
        self.mappings_ = {}
        for j in range(X.shape[1]):
            if not is_numeric_col(X[:, j]):
                self.mappings_[j] = dict(zip(X[:, j], frequency_encode(X[:, j])))
        return self

    def transform(self, X):
        X = np.array(X, dtype=object)
        for j, mapping in self.mappings_.items():
            X[:, j] = [mapping.get(value, 0.0) for value in X[:, j]]
        return X.astype(float)


class ImpactEncoderTransformer(TransformerMixin, BaseEstimator):
    """Encode string columns using class-1 rates learned during fit."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        self.default_ = float(np.mean(y))
        self.mappings_ = {}
        for j in range(X.shape[1]):
            if not is_numeric_col(X[:, j]):
                self.mappings_[j] = dict(zip(X[:, j], impact_encode(X[:, j], y)))
        return self

    def transform(self, X):
        X = np.array(X, dtype=object)
        for j, mapping in self.mappings_.items():
            X[:, j] = [mapping.get(value, self.default_) for value in X[:, j]]
        return X.astype(float)


if __name__ == "__main__":
    check_functions()

    X, y = load_credit_approval_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42
    )
    for encoder in [FrequencyEncoderTransformer(), ImpactEncoderTransformer()]:
        X_encoded = encoder.fit_transform(X_train, y_train)
        print(f"\n{type(encoder).__name__}: {X_encoded.dtype}, shape {X_encoded.shape}")
        pipe = make_pipeline(encoder, DecisionTreeClassifier(random_state=42))
        pipe.fit(X_train, y_train)
        print(f"Decision tree test accuracy: {accuracy_score(y_test, pipe.predict(X_test)):.3f}")
