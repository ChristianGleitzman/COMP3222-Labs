"""Lab 2, Task 1: Gini impurity and Gini gain on PlayGolf."""

from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "lab2" / "playgolf.csv"


def impurity(counts: np.ndarray) -> float:
    """Return Gini impurity for one node given class counts."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p**2)


def gini_gain(attr: np.ndarray, y: np.ndarray) -> float:
    """Compute Gini gain for splitting labels y by categorical attribute attr."""
    # 1. Impurity of the root, I(Z).
    _, root_counts = np.unique(y, return_counts=True)
    root_impurity = impurity(root_counts)

    # 2. Split y by each value of attr and weight each child's impurity by |W|/|Z|.
    values, value_index = np.unique(attr, return_inverse=True)
    split_impurity = 0.0
    for v in range(len(values)):
        y_child = y[value_index == v]
        _, child_counts = np.unique(y_child, return_counts=True)
        split_impurity += len(y_child) / len(y) * impurity(child_counts)

    # 3. Gini(Z, A) = I(Z) - sum of weighted child impurities.
    return root_impurity - split_impurity


@njit(cache=True)
def gini_gain_numba(attr: np.ndarray, y: np.ndarray) -> float:
    """Loop version of gini_gain. attr holds 0..k-1 and y holds 0..c-1."""
    n = attr.shape[0]
    k = attr.max() + 1
    c = y.max() + 1

    # Count cases for each (attribute value, class) pair in one pass.
    table = np.zeros((k, c))
    for i in range(n):
        table[attr[i], y[i]] += 1

    root_counts = np.zeros(c)
    for v in range(k):
        for j in range(c):
            root_counts[j] += table[v, j]

    root_sum = 0.0
    for j in range(c):
        p = root_counts[j] / n
        root_sum += p * p
    root_impurity = 1.0 - root_sum

    split_impurity = 0.0
    for v in range(k):
        n_v = 0.0
        for j in range(c):
            n_v += table[v, j]
        if n_v == 0:
            continue
        child_sum = 0.0
        for j in range(c):
            p = table[v, j] / n_v
            child_sum += p * p
        split_impurity += n_v / n * (1.0 - child_sum)

    return root_impurity - split_impurity


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    y = data.iloc[:, 4].to_numpy()
    X = data.iloc[:, 0:4].to_numpy()

    # A) Root impurity: 5 No and 9 Yes gives 1 - (5/14)^2 - (9/14)^2 = 0.4592.
    classes, counts = np.unique(y, return_counts=True)
    print("Class counts:", dict(zip(classes.tolist(), counts.tolist())))
    print(f"Root impurity: {impurity(counts):.4f}")

    # B) Outlook holds strings, so convert it to integers first.
    _, X[:, 0] = np.unique(X[:, 0], return_inverse=True)
    X = X.astype(np.int64)
    y = y.astype(np.int64)

    print(f"{'Attribute':<10}{'NumPy':>8}{'Numba':>8}")
    for j, name in enumerate(data.columns[:4]):
        print(f"{name:<10}{gini_gain(X[:, j], y):>8.4f}{gini_gain_numba(X[:, j], y):>8.4f}")
    # Expected: Outlook 0.1163, Temp 0.0163, Humidity 0.0274, Windy 0.0306.
