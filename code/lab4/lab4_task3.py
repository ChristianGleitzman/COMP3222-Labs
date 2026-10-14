"""Lab 4, Task 3: shapelet distance and convolution from scratch (student starter)."""

import matplotlib.pyplot as plt
import numpy as np

from aeon.datasets import load_gunpoint


def z_normalise(x: np.ndarray) -> np.ndarray:
    """Return (x - mean) / std. If std is zero, return an array of zeros."""
    raise NotImplementedError("Implement z_normalise")


def distance_profile(shapelet: np.ndarray, series: np.ndarray) -> np.ndarray:
    """Return the Euclidean distance between the normalised shapelet and the
    normalised window of series starting at each position.

    A series of length m and a shapelet of length L give m - L + 1 distances.
    """
    raise NotImplementedError("Implement distance_profile")


def shapelet_distance(shapelet: np.ndarray, series: np.ndarray) -> float:
    """Return sDist: the smallest value in the distance profile."""
    raise NotImplementedError("Implement shapelet_distance")


def apply_kernel(series: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Slide the kernel along the series and return the dot product at each
    position (the activation map). Do not pad the series."""
    raise NotImplementedError("Implement apply_kernel")


def max_and_ppv(series: np.ndarray, kernel: np.ndarray) -> tuple:
    """Return the maximum and the proportion of positive values (PPV) of the
    activation map."""
    raise NotImplementedError("Implement max_and_ppv")


def check_functions():
    """Compare your functions with the examples on L4.2 slides 24 and 30."""
    series = np.array([0, 3, 9, 14, 1, 2, 7, 1, 2, 1, 15, 2, 10], dtype=float)
    shapelet = np.array([-5, 0, 5], dtype=float)
    print("Slide 24, first window normalised:", np.round(z_normalise(series[:3]), 3))
    print("Slide 24, distance profile:", np.round(distance_profile(shapelet, series), 3))
    print("Slide 24, sDist:", round(shapelet_distance(shapelet, series), 3))

    series = np.array([1, 2, 4, 3, 1], dtype=float)
    kernel = np.array([-1, 0, 1], dtype=float)
    print("Slide 30, activation map:", apply_kernel(series, kernel))
    print("Slide 30, max and PPV:", max_and_ppv(series, kernel))


# Optional extension: shapelet quality on the order line (L4.2 slides 25 to 27).

def entropy(y: np.ndarray) -> float:
    """Return the entropy, in bits, of the class labels y."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def information_gains(distances: np.ndarray, y: np.ndarray) -> list:
    """Return (split_point, information_gain) for every split of the order line.

    Sort the distances, then try each split point halfway between consecutive
    distances. Use entropy() for the root and for both sides of the split.
    """
    raise NotImplementedError("Implement information_gains")


if __name__ == "__main__":
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    print(f"GunPoint train {X_train.shape}, test {X_test.shape}")

    shapelet = X_train[5, 0, 100:140]
    kernel = None  # TODO: use the z-normalised shapelet as a kernel.
    print(f"Shapelet taken from training case 5 (class {y_train[5]}), timepoints 100 to 139")

    # TODO: Implement the functions above, then call check_functions().
    # TODO: Pick one training case from each class. Plot, in three rows with
    #       plt.subplots(3, 1, sharex=True): the two series, their distance
    #       profiles and their activation maps.
    # TODO: Scatter plot sDist against max activation for every training case,
    #       coloured by class. Which feature separates the classes?
