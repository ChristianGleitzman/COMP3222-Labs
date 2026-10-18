"""Lab 4, Task 1: time series data, distances and nearest neighbours (student starter)."""

import math

import matplotlib.pyplot as plt
import numpy as np
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.datasets import load_basic_motions, load_gunpoint
from numba import njit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier


def plot_examples(X, y, n_per_class=3):
    """Plot the first channel of a few cases from each class."""
    for label, colour in zip(np.unique(y), ["tab:blue", "tab:red", "tab:green", "tab:orange"]):
        for series in X[y == label][:n_per_class]:
            plt.plot(series[0], color=colour, alpha=0.7)
        plt.plot([], [], color=colour, label=f"class {label}")
    plt.xlabel("Timepoint")
    plt.legend()
    plt.show()

@njit(cache=True)
def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Return the Euclidean distance between two equal-length 1D series."""
    # You wrote this in Lab 1. Add @njit(cache=True) from numba for speed.
    squared_sum = 0
    for i in range(len(x)):
        squared_sum += (x[i] - y[i]) ** 2
    return math.sqrt(squared_sum)

@njit(cache=True)
def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Return the DTW cost between two 1D series.

    Use squared pointwise differences and do not take a square root at the end,
    so that the result matches aeon.distances.dtw_distance.
    """
    # See the Lab 1 optional extension and code/lab1/lab1_task1_solution.py.
    m = len(x)
    costs = np.full((m + 1, m + 1), np.inf)
    costs[0, 0] = 0.0

    for i in range(1, m + 1):
        for j in range(1, m + 1):
            difference = x[i - 1] - y[j - 1]
            costs[i, j] = difference**2 + min(
                costs[i - 1, j - 1], costs[i - 1, j], costs[i, j - 1]
            )
    return costs[m, m]


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
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        func = euclidean_distance if self.distance == "euclidean" else dtw_distance
        predictions = []

        for case in X:
            minimum = np.inf
            best = None
            for train_case, label in zip(self.X_, self.y_):
                curr_distance = func(case[0], train_case[0])
                if curr_distance < minimum:
                    minimum = curr_distance
                    best = label
            predictions.append(best)

        return np.array(predictions)





if __name__ == "__main__":
    X_train, y_train = load_gunpoint(split="train") #type: ignore
    X_test, y_test = load_gunpoint(split="test") #type: ignore
    X_train, y_train, X_test, y_test = map(np.asarray, (X_train, y_train, X_test, y_test))
    print(f"GunPoint train: X {X_train.shape}, y {y_train.shape}")
    print(f"GunPoint test:  X {X_test.shape}, y {y_test.shape}")

    X_motion, y_motion = map(np.asarray, load_basic_motions(split="train"))
    print(f"BasicMotions train: X {X_motion.shape}, classes {np.unique(y_motion)}")

    # TODO: Use plot_examples to look at both problems.
    # TODO: Implement the distances, then call check_distances().
    # TODO: Fit RandomForestClassifier on X_train[:, 0, :] and record test accuracy.
    # TODO: Shuffle the timepoints with one permutation for train and test, then repeat.
    # TODO: Compare OneNearestNeighbour with "euclidean" and "dtw", before and after
    #       shuffling, and check against KNeighborsTimeSeriesClassifier in aeon.
    print(X_train[3,0]) # case 3
    check_distances()
    #plot_examples(X_train, y_train)
    #plot_examples(X_motion, y_motion)

    onn = OneNearestNeighbour()
    onn.fit(X_train, y_train)
    print(f"1NN score: {onn.score(X_test, y_test)}")

    dtw = OneNearestNeighbour(distance="dtw").fit(X_train, y_train)
    print("1-NN dtw accuracy:", dtw.score(X_test, y_test))

    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(X_train[:, 0, :], y_train)
    print("RF accuracy:", rfc.score(X_test[:, 0, :], y_test))

    # Shuffle the timepoints with one permutation shared by train and test.
    rng = np.random.default_rng(0)
    perm = rng.permutation(X_train.shape[2])
    X_train_s, X_test_s = X_train[:, :, perm], X_test[:, :, perm]

    results = {}
    for name, Xtr, Xte in [("original", X_train, X_test), ("shuffled", X_train_s, X_test_s)]:
        rf = RandomForestClassifier(random_state=0).fit(Xtr[:, 0, :], y_train)
        results[(name, "RF")] = rf.score(Xte[:, 0, :], y_test)
        for dist in ["euclidean", "dtw"]:
            mine = OneNearestNeighbour(distance=dist).fit(Xtr, y_train)
            results[(name, f"1NN-{dist}")] = mine.score(Xte, y_test)
            aeon_knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance=dist)
            aeon_knn.fit(Xtr, y_train)
            results[(name, f"aeon-{dist}")] = aeon_knn.score(Xte, y_test)

    print("\nGunPoint test accuracy")
    for (name, model), acc in results.items():
        print(f"{name:9s} {model:14s} {acc:.4f}")

