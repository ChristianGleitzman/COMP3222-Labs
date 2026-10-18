"""Lab 4, Task 2: summary features and random intervals (student starter)."""

import numpy as np
from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.datasets import load_basic_motions, load_gunpoint
from aeon.transformations.collection.feature_based import Catch22
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline


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
    # A single ':' in X[:, 1] for example, means "everything in this axis". Same idea as splicing just the general, unrestricted form

    means = X.mean(axis=2)
    stds = X.std(axis=2)
    mins = X.min(axis=2)
    maxes = X.max(axis=2)
    slopes = slope(X)
    return np.stack([means,stds,mins,maxes,slopes], axis = 2).reshape(len(X),-1)


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
        return self

    def transform(self, X):
        return summary_features(X)


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
        rng = np.random.default_rng(self.random_state)
        series_length = X.shape[-1]
        self.intervals_ = []
        if self.min_length > series_length:
            raise IndexError("Minimum Interval Length is longer than the Series Length")
        for _ in range(self.n_intervals):
            start = rng.integers(0, series_length - self.min_length + 1)
            end = rng.integers(start + self.min_length, series_length + 1)
            self.intervals_.append((start,end))
        return self

    def transform(self, X):
        # TODO: For each interval, X[:, :, start:end] gives one subseries per
        # channel. Concatenate its mean, std and slope into a 2D array.
        features = []
        for interval in self.intervals_:
            curr = X[:, :, interval[0]:interval[1]]
            means = curr.mean(axis=-1)
            stds = curr.std(axis=-1)
            slopes = slope(curr)
            curr_transform = np.stack([means,stds,slopes], axis = -1).reshape(len(curr),-1)
            features.append(curr_transform)

        return np.concatenate(features, axis=1)

if __name__ == "__main__":
    X_train, y_train = load_gunpoint(split="train") #type: ignore
    X_test, y_test = load_gunpoint(split="test") #type: ignore
    X_motion, y_motion = load_basic_motions(split="train") #type: ignore
    print(f"GunPoint train {X_train.shape}; BasicMotions train {X_motion.shape}") #type: ignore

    # TODO: Implement summary_features, call check_functions() and print the
    #       feature shapes for GunPoint and BasicMotions.
    # TODO: Compare RandomForestClassifier on the raw series, on summary
    #       features and on random interval features (use make_pipeline).
    # TODO: Try Catch22 + RandomForestClassifier and TimeSeriesForestClassifier.

    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train.reshape(len(X_train), -1), y_train)
    print("RF raw accuracy:", rf.score(X_test.reshape(len(X_test), -1), y_test))

    rf_summary = make_pipeline(SummaryTransformer(), RandomForestClassifier(random_state=0))
    rf_summary.fit(X_train, y_train)
    print("RF summary accuracy:", rf_summary.score(X_test, y_test))

    rf_interval = make_pipeline(RandomIntervalTransformer(), RandomForestClassifier(random_state=0))
    rf_interval.fit(X_train, y_train)
    print("RF interval accuracy:", rf_interval.score(X_test, y_test))

    tsf = TimeSeriesForestClassifier(n_estimators=200, random_state=0)
    tsf.fit(X_train, y_train)
    print(tsf.score(X_test, y_test))

    aeon_cls = make_pipeline(Catch22(), RandomForestClassifier(random_state=0))
    aeon_cls.fit(X_train, y_train)
    print("Catch 22 accuracy:", rf_interval.score(X_test, y_test))
