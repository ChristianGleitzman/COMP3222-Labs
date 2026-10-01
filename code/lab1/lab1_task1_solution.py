"""Lab 1, Task 1: Euclidean distance, Numba, and optional DTW."""

import math
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit


def euclidean_distance_no_numba(x: np.ndarray, y: np.ndarray) -> float:
    """Sum squared differences in one loop, then take the square root."""
    total = 0.0
    for i in range(len(x)):
        difference = x[i] - y[i]
        total += difference * difference
    return math.sqrt(total)


@njit(cache=True)
def euclidean_distance_numba(x: np.ndarray, y: np.ndarray) -> float:
    """The same calculation, compiled by Numba."""
    total = 0.0
    for i in range(len(x)):
        difference = x[i] - y[i]
        total += difference * difference
    return math.sqrt(total)


def dtw_distance_no_numba(x: np.ndarray, y: np.ndarray) -> float:
    """Return the accumulated squared DTW cost from the lab-sheet algorithm."""
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


@njit(cache=True)
def dtw_distance_numba(x: np.ndarray, y: np.ndarray) -> float:
    """The same DTW calculation, compiled by Numba."""
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


def timing_experiment(max_n: int, increment: int, function1, function2):
    """Time both functions on the same random arrays; return times in seconds.

    Warm up a Numba function before calling this, so compilation is not timed.
    A single run per size is simple but can be noisy for very fast functions.
    """
    rng = np.random.default_rng(0)
    sizes, t1, t2 = [], [], []
    for n in range(increment, max_n + 1, increment):
        x, y = rng.random(n), rng.random(n)

        start = time.perf_counter()
        result1 = function1(x, y)
        t1.append(time.perf_counter() - start)

        start = time.perf_counter()
        result2 = function2(x, y)
        t2.append(time.perf_counter() - start)

        if not np.isclose(result1, result2):
            raise AssertionError(f"Results differ at n={n}: {result1} != {result2}")
        sizes.append(n)
    return np.array(sizes), np.array(t1), np.array(t2)


def plot_timing(sizes, t1, t2, title):
    """Plot input length against elapsed seconds on log-log axes."""
    plt.figure(figsize=(7, 5))
    plt.plot(sizes, t1, marker="o", label="Python")
    plt.plot(sizes, t2, marker="s", label="Numba")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Series length n (log scale)")
    plt.ylabel("Time in seconds (log scale)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 3.0, 4.0])

    # The first Numba call compiles each function. Leave it out of the timings.
    assert np.isclose(euclidean_distance_no_numba(x, y), euclidean_distance_numba(x, y))
    assert np.isclose(dtw_distance_no_numba(x, y), dtw_distance_numba(x, y))

    sizes, python_times, numba_times = timing_experiment(
        1_000_000, 100_000, euclidean_distance_no_numba, euclidean_distance_numba
    )
    plot_timing(sizes, python_times, numba_times, "Euclidean distance: Python vs Numba")
    print(f"Euclidean speed-up at n={sizes[-1]}: {python_times[-1] / numba_times[-1]:.1f}x")

    # DTW is O(n²), so use much smaller arrays.
    sizes, python_times, numba_times = timing_experiment(
        500, 100, dtw_distance_no_numba, dtw_distance_numba
    )
    plot_timing(sizes, python_times, numba_times, "DTW cost: Python vs Numba")
    print(f"DTW speed-up at n={sizes[-1]}: {python_times[-1] / numba_times[-1]:.1f}x")
    plt.show()
