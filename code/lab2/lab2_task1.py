from pathlib import Path

import pandas as pd
import numpy as np

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "lab2" / "playgolf.csv"

def impurity(counts: np.ndarray) -> float:
    """Return Gini Impurity for one node given class counts"""
    total = counts.sum()
    if total == 0: return 0.0
    imp = 1 - np.sum((counts / total) ** 2)
    return imp

def gini_gain(attr: np.ndarray, y: np.ndarray) -> float:
    """Compute Gini gain for splitting labels y by categorical
    attribute attr."""

    classes, root_counts = np.unique_counts(y)
    I_root = impurity(root_counts)

    sum = 0
    for v in np.unique(attr):
        mask = attr == v
        counts = np.array([(y[mask] == c).sum() for c in classes])
        sum += (mask.sum() / len(y)) * impurity(counts)
    return I_root - sum

if __name__ == "__main__":    # Example usage
    data = pd.read_csv(DATA_PATH)
    print(data)
    y = data.iloc[:, 4].to_numpy()
    print(y)
    X = data.iloc[:, 0:4].to_numpy()
    print(X.shape)
    outlook = X[:, 0]
    print(outlook)
    print(outlook.shape)
