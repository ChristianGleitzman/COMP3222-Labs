"""Lab 2, Task 1: Gini impurity and gain on the PlayGolf data.

The first two functions match the signatures in the lab sheet. The Numba
versions at the end are an optional way to repeat the calculation with loops.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "lab2" / "playgolf.csv"


def impurity(counts: np.ndarray) -> float:
    """Return 1 - sum(p²) for the class counts at one node."""
    counts = np.asarray(counts)
    total = counts.sum()
    if total == 0:
        return 0.0
    proportions = counts / total
    return float(1 - np.sum(proportions**2))


def gini_gain(attr: np.ndarray, y: np.ndarray) -> float:
    """Return the reduction in impurity from splitting y by attr.

    Each child node corresponds to one distinct attribute value. Its impurity
    is weighted by the fraction of rows that enter that child.
    """
    attr, y = np.asarray(attr), np.asarray(y)
    if attr.ndim != 1 or y.ndim != 1 or len(attr) != len(y):
        raise ValueError("attr and y must be 1D arrays of the same length")
    if len(y) == 0:
        return 0.0

    root_counts = np.unique(y, return_counts=True)[1]
    weighted_children = 0.0
    for value in np.unique(attr):
        child_y = y[attr == value]
        child_counts = np.unique(child_y, return_counts=True)[1]
        weighted_children += len(child_y) / len(y) * impurity(child_counts)
    return impurity(root_counts) - weighted_children


@njit(cache=True)
def impurity_numba(counts: np.ndarray) -> float:
    """Optional loop version of impurity for integer count arrays."""
    total = 0
    for count in counts:
        total += count
    if total == 0:
        return 0.0

    squared_proportions = 0.0
    for count in counts:
        proportion = count / total
        squared_proportions += proportion * proportion
    return 1.0 - squared_proportions


@njit(cache=True)
def gini_gain_numba(attr: np.ndarray, y: np.ndarray) -> float:
    """Optional loop version; attr and y must be 0-based integer codes."""
    n = len(y)
    if n == 0:
        return 0.0

    n_values = np.max(attr) + 1
    n_classes = np.max(y) + 1
    root_counts = np.zeros(n_classes, dtype=np.int64)
    child_counts = np.zeros((n_values, n_classes), dtype=np.int64)
    group_sizes = np.zeros(n_values, dtype=np.int64)

    for i in range(n):
        value, label = attr[i], y[i]
        root_counts[label] += 1
        child_counts[value, label] += 1
        group_sizes[value] += 1

    weighted_children = 0.0
    for value in range(n_values):
        if group_sizes[value] > 0:
            weighted_children += group_sizes[value] / n * impurity_numba(
                child_counts[value]
            )
    return impurity_numba(root_counts) - weighted_children


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    y = data["PlayGolf"].to_numpy()
    root_counts = np.unique(y, return_counts=True)[1]
    print(f"Root Gini impurity: {impurity(root_counts):.4f}")

    for column in data.columns[:-1]:
        attr = data[column].to_numpy()
        gain = gini_gain(attr, y)
        print(f"{column:8s} Gini gain: {gain:.4f}")

        # Numba needs integer codes, including for the string Outlook values.
        codes = np.unique(attr, return_inverse=True)[1].astype(np.int64)
        assert np.isclose(gain, gini_gain_numba(codes, y.astype(np.int64)))
    print("The optional Numba version gives the same gains.")
