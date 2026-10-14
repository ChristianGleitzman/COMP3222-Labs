"""Lab 4, Task 2: summary features and random intervals (student starter)."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from aeon.datasets import load_basic_motions, load_gunpoint


def slope(X: np.ndarray) -> np.ndarray:
    """Return the least-squares slope along the last axis.

    For X of shape (n_cases, n_channels, n_timepoints) the result has shape
    (n_cases, n_channels): one slope per channel of each case.
    """
    t = np.arange(X.shape[-1], dtype=float)
    t -= t.mean()
    centred = X - X.mean(axis=-1, keepdims=True)
    return (centred * t).sum(axis=-1) / (t ** 2).sum()


def summary_features(X: np.ndarray) -> np.ndarray:
    """Return mean, std, min, max and slope of every channel as a 2D array.

    X has shape (n_cases, n_channels, n_timepoints); the result should have
    shape (n_cases, 5 * n_channels). See L4.2 slide 9.
    """
    raise NotImplementedError("Implement summary_features")


def check_functions():
    """Print features for the two series on L4.2 slides 8 and 18."""
    X = np.array([[[0, 0, 2, 2]], [[0, 2, 0, 2]]], dtype=float)
    print("     mean   std   min   max  slope")
    for name, row in zip("ab", summary_features(X)):
        print(name, " ".join(f"{value:5.2f}" for value in row))


class SummaryTransformer(TransformerMixin, BaseEstimator):
    """Wrap summary_features as a scikit-learn transformer."""

    def fit(self, X, y=None):
        # Nothing needs to be learned, but fit must still return self.
        raise NotImplementedError("Implement SummaryTransformer.fit")

    def transform(self, X):
        raise NotImplementedError("Implement SummaryTransformer.transform")


class RandomIntervalTransformer(TransformerMixin, BaseEstimator):
    """Mean, std and slope of random intervals, as in Time Series Forest.

    In fit, choose n_intervals (start, end) pairs, each at least min_length
    timepoints long. In transform, use the same intervals for every case.
    """

    def __init__(self, n_intervals=10, min_length=3, random_state=None):
        self.n_intervals = n_intervals
        self.min_length = min_length
        self.random_state = random_state

    def fit(self, X, y=None):
        # TODO: rng = np.random.default_rng(self.random_state)
        # TODO: Store the chosen intervals in self.intervals_.
        raise NotImplementedError("Implement RandomIntervalTransformer.fit")

    def transform(self, X):
        # TODO: For each interval, X[:, :, start:end] gives one subseries per
        # channel. Concatenate its mean, std and slope into a 2D array.
        raise NotImplementedError("Implement RandomIntervalTransformer.transform")


if __name__ == "__main__":
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    X_motion, y_motion = load_basic_motions(split="train")
    print(f"GunPoint train {X_train.shape}; BasicMotions train {X_motion.shape}")

    # TODO: Implement summary_features, call check_functions() and print the
    #       feature shapes for GunPoint and BasicMotions.
    # TODO: Compare RandomForestClassifier on the raw series, on summary
    #       features and on random interval features (use make_pipeline).
    # TODO: Try Catch22 + RandomForestClassifier and TimeSeriesForestClassifier.
