"""Lab 4, Task 3: shapelet distance and convolution from scratch (student starter)."""

import time

import matplotlib.pyplot as plt
import numpy as np

from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.shapelet_based import RDSTClassifier
from aeon.datasets import load_gunpoint


def z_normalise(x: np.ndarray) -> np.ndarray:
    """Return (x - mean) / std. If std is zero, return an array of zeros."""
    x_std = x.std()
    if x_std == 0:
        return np.zeros_like(x, dtype=float)
    else:
        return (x - x.mean()) / x_std


def distance_profile(shapelet: np.ndarray, series: np.ndarray) -> np.ndarray:
    """Return the Euclidean distance between the normalised shapelet and the
    normalised window of series starting at each position.

    A series of length m and a shapelet of length L give m - L + 1 distances.
    """
    norm_shapelet = z_normalise(shapelet)

    m, L = series.shape[0], shapelet.shape[0]
    distances = []
    for start in range(m - L + 1):
        norm_window = z_normalise(series[start:start+L])
        distances.append(np.linalg.norm(norm_window - norm_shapelet))
    return np.asarray(distances)


def shapelet_distance(shapelet: np.ndarray, series: np.ndarray) -> float:
    """Return sDist: the smallest value in the distance profile."""
    return distance_profile(shapelet, series).min()


def apply_kernel(series: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Slide the kernel along the series and return the dot product at each
    position (the activation map). Do not pad the series."""
    activation_map = []
    m, L = series.shape[0], kernel.shape[0]
    for start in range(m - L + 1):
        prod = np.dot(series[start:start+L], kernel)
        activation_map.append(prod)
    return np.asarray(activation_map)


def max_and_ppv(series: np.ndarray, kernel: np.ndarray) -> tuple:
    """Return the maximum and the proportion of positive values (PPV) of the
    activation map."""
    act_map = apply_kernel(series, kernel)

    return act_map.max(), np.mean(act_map > 0)


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
    order = np.argsort(distances)
    sorted_d, sorted_y = distances[order], y[order]
    root = entropy(y)
    n = len(y)
    gains = []
    for i in range(1, n):
        if sorted_d[i] == sorted_d[i - 1]:
            continue  # no threshold separates equal distances
        split = (sorted_d[i - 1] + sorted_d[i]) / 2
        left, right = sorted_y[:i], sorted_y[i:]
        weighted = (len(left) * entropy(left) + len(right) * entropy(right)) / n
        gains.append((split, root - weighted))
    return gains


if __name__ == "__main__":
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    print(f"GunPoint train {X_train.shape}, test {X_test.shape}")

    shapelet = X_train[5, 0, 100:140]
    kernel = z_normalise(shapelet)
    print(f"Shapelet taken from training case 5 (class {y_train[5]}), timepoints 100 to 139")

    check_functions()

    # One training case from each class: series, distance profile, activation map.
    classes = np.unique(y_train)
    colours = ["tab:blue", "tab:red"]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
    for label, colour in zip(classes, colours):
        series = X_train[y_train == label][0, 0]
        axes[0].plot(series, color=colour, label=f"class {label}")
        axes[1].plot(distance_profile(shapelet, series), color=colour)
        axes[2].plot(apply_kernel(series, kernel), color=colour)
    axes[0].set_ylabel("Series")
    axes[1].set_ylabel("Distance profile")
    axes[2].set_ylabel("Activation map")
    axes[2].set_xlabel("Start position")
    axes[0].legend()
    plt.tight_layout()
    plt.show()

    # sDist against max activation for every training case.
    s_dists = np.array([shapelet_distance(shapelet, case[0]) for case in X_train])
    max_acts = np.array([max_and_ppv(case[0], kernel)[0] for case in X_train])
    for label, colour in zip(classes, colours):
        mask = y_train == label
        plt.scatter(s_dists[mask], max_acts[mask], color=colour, label=f"class {label}")
        print(f"class {label}: mean sDist {s_dists[mask].mean():.3f}, "
              f"mean max activation {max_acts[mask].mean():.3f}")
    plt.xlabel("sDist")
    plt.ylabel("Max activation")
    plt.legend()
    plt.show()

    # Optional extension: best information gain on the order line of each feature.
    for name, feature in [("sDist", s_dists), ("max activation", max_acts)]:
        split, gain = max(information_gains(feature, y_train), key=lambda sg: sg[1])
        print(f"{name}: best split {split:.3f}, information gain {gain:.3f} bits")

    # ROCKET (random kernels) against RDST (random dilated shapelets) on GunPoint.
    for name, clf in [
        ("Rocket", RocketClassifier(n_kernels=2000, random_state=0)),
        ("RDST", RDSTClassifier(random_state=0)),
    ]:
        start = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        print(f"{name}: test accuracy {clf.score(X_test, y_test):.4f}, fit time {fit_time:.1f}s")
