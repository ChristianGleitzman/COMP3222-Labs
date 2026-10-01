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
from numba import njit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Data lives in <repo root>/data/lab1, this file is in <repo root>/code/lab1.
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "lab1"


def euclidean_distance_python(x: np.ndarray, y: np.ndarray) -> float:
    # if x.shape != y.shape:
    #     raise ValueError
    squared_sum = 0
    for i in range(len(x)):
        squared_sum += (x[i] - y[i]) ** 2
    return math.sqrt(squared_sum)

@njit(cache=True)
def euclidean_distance_numba(x: np.ndarray, y: np.ndarray) -> float:
    squared_sum = 0
    for i in range(len(x)):
        squared_sum += (x[i] - y[i]) ** 2
    return math.sqrt(squared_sum)


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
    t_numba_list = []
    for n in range(100_000, 1_000_001, 100_000):
        sizes.append(n)
        t_py_list.append(time_distance(euclidean_distance_python, n))
        t_numba_list.append(time_distance(euclidean_distance_numba, n))
        print(f"n = {n:>9,}  python = {t_py_list[-1]:.6f}s")
        print(f"n = {n:>9,}  numba = {t_numba_list[-1]:.6f}s")

    # --- Plot: time vs n (log-log) ---
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(sizes, t_py_list, marker="o", label="ED Python")
    plt.plot(sizes, t_numba_list, marker="o", label="ED Numba")
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



    # One-hot encoding of colour
    encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore') # Sparse output makes the encoder return a dense array
    # Encoding a single feature needs shape (-1,1) for 2D formatting
    # fit requires reshaping
    # fit_transform does the fit and transformation of the data for you
    colour_onehot = encoder.fit_transform(colour.reshape(-1,1))
    # Using column stack to append the whole rank feature column as a float
    X = np.column_stack((X, rank.astype(float), colour_onehot))

    # Creating a dummy classifier to ensure X is clean
    dc = DummyClassifier()
    dc.fit(X, y)

    print(X.shape, rank.shape, colour.shape, y.shape)
    print(type(X), type(rank), type(colour), type(y))
    return X, rank, colour, y


def create_column_pipeline(clf=None) -> Pipeline:
    """Build a pipeline that preprocesses the mixed one_hot.csv columns.

    Continuous and ordinal columns are imputed and standardised, and the nominal
    ``colour`` column is one-hot encoded. The pipeline takes the raw DataFrame
    (without ``y``) as input.

    Parameters
    ----------
    clf : estimator, default=None
        Classifier for the final step. If None, uses ``KNeighborsClassifier()``.

    Returns
    -------
    Pipeline
        The unfitted preprocessing + classifier pipeline.
    """
    if clf is None:
        clf = KNeighborsClassifier()
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric, ["x1", "x2", "x3", "rank"]),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["colour"]),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("clf", clf),
    ])


def pipeline_example():
    """Fit ``create_column_pipeline`` on one_hot.csv and report test accuracy."""
    df = pd.read_csv(DATA_DIR / "one_hot.csv", header=None,
                     names=["x1", "x2", "x3", "rank", "colour", "y"])
    X = df.drop(columns="y")
    y = df["y"].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    pipe = create_column_pipeline()
    pipe.fit(X_train, y_train)
    print(f"Pipeline test accuracy: {pipe.score(X_test, y_test):.3f}")
    return pipe


class SingleFeatureClassifier(ClassifierMixin, BaseEstimator):
    """Task 3: binary classifier using a threshold on a single feature.

    Parameters
    ----------
    feature : int, default=0
        Index of the column of X to use.

    Attributes
    ----------
    classes_ : np.ndarray
        The two class labels seen in ``fit``.
    threshold_ : float
        Median of the chosen feature in the training data.
    less_than_class_ : int
        Class predicted when ``value < threshold_``.
    greater_class_ : int
        Class predicted otherwise.
    """

    def __init__(self, feature: int = 0):
        # Constructor only stores its arguments, by sklearn convention.
        self.feature = feature

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SingleFeatureClassifier only handles binary problems.")

        col = X[:, self.feature]
        means = [np.average(col[y == c]) for c in self.classes_]
        self.threshold_ = np.median(col)
        self.less_than_class_ = self.classes_[np.argmin(means)]
        self.greater_than_class_ = self.classes_[np.argmax(means)]

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        col = X[:, self.feature]
        return np.where(col < self.threshold_, self.less_than_class_, self.greater_than_class_)


def task3_example():
    """Test SingleFeatureClassifier on the imputed missing.csv data."""
    X, y = missing_example()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", SingleFeatureClassifier(feature=0)),
    ])
    pipe.fit(X_train, y_train)
    print(f"SingleFeatureClassifier test accuracy: {pipe.score(X_test, y_test):.3f}")


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

    si = SimpleImputer(missing_values=np.nan, strategy='mean') # Make sure you define what a "missing" value actually means!
    X_filled = si.fit_transform(X) # Simple imputer can employ simple strategies, like mean!

    print(f"X shape {X.shape}, {np.isnan(X).sum()} missing values")
    print(f"X_filled shape {X.shape}, {np.isnan(X_filled).sum()} missing values")

    ss = StandardScaler()
    X_scaled = ss.fit_transform(X_filled)
    return X_scaled, y

def create_standard_pipeline():
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ])
    return pipe # completes all steps automatically :)

def standard_pipeline_example():
    X, y = missing_example()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,random_state=42,shuffle=True
    )
    pipe = create_standard_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)


if __name__ == "__main__":
    one_hot_example()
    missing_example()
    pipeline_example()
    task3_example()
    #part1_example(False)
