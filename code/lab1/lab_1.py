"""Example code for Lab Sheet 1.

This file is the worked example referred to in the lab sheet. It shows

* Task 1: a pure Python Euclidean distance, and the timing/plotting pattern you
  need for the timing experiment.
* Task 2: how to load the two data files in ``data/lab1`` with pandas.

The exercises themselves (the Numba version, DTW, the preprocessing pipeline and
the single feature classifier) are left for you to write.
"""

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data lives in <repo root>/data/lab1, this file is in <repo root>/code/lab1.
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "lab1"


def euclidean_distance_python(x: np.ndarray, y: np.ndarray) -> float:
    """Add a docstring."""
    return 0



def time_distance(distance_function, n: int, random_state: int = 0) -> float:
    """Time a single call of ``distance_function`` on two random series of length n.

    Parameters
    ----------
    distance_function : callable
        Takes two 1D np.ndarray and returns a float.
    n : int
        Length of the random series to generate.
    random_state : int, default=0
        Seed for the random number generator, so runs are repeatable.

    Returns
    -------
    float
        Time taken in seconds.
    """
    rng = np.random.default_rng(random_state)
    x = rng.random(n)
    y = rng.random(n)
    start = time.perf_counter()
    distance_function(x, y)
    return time.perf_counter() - start


def part1_example(show: bool = True):
    """Time the Python distance for increasing n and plot the result.

    This is the timing and plotting pattern for Task 1.3. It only times the
    pure Python version. Once you have written your ``@njit`` version, time it
    in the same loop and add a second line to the plot, so you can compare the
    two and estimate the speed-up.

    Parameters
    ----------
    show : bool, default=True
        If True, display the plot. The figure is saved either way.

    Returns
    -------
    tuple of list
        The series lengths used and the times taken.
    """
    # Basic usage.
    x = np.arange(1.0, 11.0)
    y = np.arange(2.0, 12.0)
    print("Python:", euclidean_distance_python(x, y))

    # Time for increasing sizes of input.
    sizes = []
    t_py_list = []
    for n in range(100_000, 1_000_001, 100_000):
        sizes.append(n)
        t_py_list.append(time_distance(euclidean_distance_python, n))
        print(f"n = {n:>9,}  python = {t_py_list[-1]:.6f}s")

    # --- Plot: time vs n (log-log) ---
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(sizes, t_py_list, marker="o", label="ED Python")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n (log scale)")
    plt.ylabel("Time (seconds, log scale)")
    plt.title("Euclidean distance: time vs n")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("jit_speed_time_vs_n.png", dpi=150)
    if show:
        plt.show()
    return sizes, t_py_list


def one_hot_example():
    """Load one_hot.csv, the mixed categorical/continuous data used in Task 2A.

    Returns
    -------
    tuple
        ``(X, rank, colour, y)``, the continuous features, the ordinal feature,
        the nominal feature and the class labels.
    """
    path = DATA_DIR / "one_hot.csv"
    df = pd.read_csv(path, header=None, names=["x1", "x2", "x3", "rank", "colour", "y"])
    X = df[["x1", "x2", "x3"]].to_numpy(dtype=float)  # shape (n, 3)
    rank = df["rank"].to_numpy(dtype=int)  # shape (n,)
    colour = df["colour"].to_numpy(dtype=str)  # shape (n,)
    y = df["y"].to_numpy(dtype=int)  # shape (n,)
    print(X.shape, rank.shape, colour.shape, y.shape)
    print(type(X), type(rank), type(colour), type(y))
    return X, rank, colour, y


def missing_example():
    """Load missing.csv, the data with missing values used in Task 2B.

    Missing values are either blank or the string ``NaN``, both of which pandas
    reads as ``np.nan``.

    Returns
    -------
    tuple of np.ndarray
        ``(X, y)``, the features (containing NaNs) and the class labels.
    """
    path = DATA_DIR / "missing.csv"
    df = pd.read_csv(path, header=None, names=["x1", "x2", "x3", "y"])
    X = df[["x1", "x2", "x3"]].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    print(f"X shape {X.shape}, {np.isnan(X).sum()} missing values")
    return X, y


if __name__ == "__main__":
    one_hot_example()
    missing_example()
    part1_example()
