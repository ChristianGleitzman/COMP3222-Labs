"""Lab 4, Task 1: time series data, distances and nearest neighbours (student starter)."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from aeon.datasets import load_basic_motions, load_gunpoint


def plot_examples(X, y, n_per_class=3):
    """Plot the first channel of a few cases from each class."""
    for label, colour in zip(np.unique(y), ["tab:blue", "tab:red", "tab:green", "tab:orange"]):
        for series in X[y == label][:n_per_class]:
            plt.plot(series[0], color=colour, alpha=0.7)
        plt.plot([], [], color=colour, label=f"class {label}")
    plt.xlabel("Timepoint")
    plt.legend()
    plt.show()


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Return the Euclidean distance between two equal-length 1D series."""
    # You wrote this in Lab 1. Add @njit(cache=True) from numba for speed.
    raise NotImplementedError("Implement euclidean_distance")


def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Return the DTW cost between two 1D series.

    Use squared pointwise differences and do not take a square root at the end,
    so that the result matches aeon.distances.dtw_distance.
    """
    # See the Lab 1 optional extension and code/lab1/lab1_task1_solution.py.
    # C = np.full((n + 1, m + 1), np.inf); C[0, 0] = 0.0
    raise NotImplementedError("Implement dtw_distance")


def check_distances():
    """Compare your distances with aeon on the pair from L4.1 slide 22."""
    from aeon.distances import dtw_distance as aeon_dtw
    from aeon.distances import euclidean_distance as aeon_euclidean

    a = np.array([1, 3, 2, 3, 15, 7, 4, 3, 1], dtype=float)
    b = np.array([2, 3, 1, 4, 4, 15, 7, 3, 1], dtype=float)
    print(f"Euclidean: yours {euclidean_distance(a, b):.3f}, aeon {aeon_euclidean(a, b):.3f}")
    print(f"DTW:       yours {dtw_distance(a, b):.3f}, aeon {aeon_dtw(a, b):.3f}")


class OneNearestNeighbour(ClassifierMixin, BaseEstimator):
    """1-NN classifier for univariate collections of shape (n_cases, 1, n_timepoints).

    distance is "euclidean" or "dtw".
    """

    def __init__(self, distance="euclidean"):
        self.distance = distance

    def fit(self, X, y):
        # TODO: Store the training series, labels and classes_.
        raise NotImplementedError("Implement OneNearestNeighbour.fit")

    def predict(self, X):
        # TODO: For each case in X, find the closest training case using the
        # chosen distance on channel 0 and return its label.
        raise NotImplementedError("Implement OneNearestNeighbour.predict")


if __name__ == "__main__":
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    print(f"GunPoint train: X {X_train.shape}, y {y_train.shape}")
    print(f"GunPoint test:  X {X_test.shape}, y {y_test.shape}")

    X_motion, y_motion = load_basic_motions(split="train")
    print(f"BasicMotions train: X {X_motion.shape}, classes {np.unique(y_motion)}")

    # TODO: Use plot_examples to look at both problems.
    # TODO: Implement the distances, then call check_distances().
    # TODO: Fit RandomForestClassifier on X_train[:, 0, :] and record test accuracy.
    # TODO: Shuffle the timepoints with one permutation for train and test, then repeat.
    # TODO: Compare OneNearestNeighbour with "euclidean" and "dtw", before and after
    #       shuffling, and check against KNeighborsTimeSeriesClassifier in aeon.
