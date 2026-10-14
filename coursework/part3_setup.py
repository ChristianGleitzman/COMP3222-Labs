"""Shared data, classifier pool and settings for Part 3 of the coursework.

Import from this module rather than copying it, so that every submission is
run on the same datasets, with the same pool and the same settings::

    from part3_setup import (CV, DATASETS, RANDOM_STATE, load_dataset,
                             make_pool, make_stacking)

The datasets are included in the course repository under ``data/ucr``.
They are read from the directory named by the environment variable
COMP3222_UCR_DIR if it is set, and from ``data/ucr`` next to this file
otherwise. A dataset that is not found there is downloaded from the UCR
archive and saved in that directory, which needs an internet connection.
"""

import os
from pathlib import Path

import numpy as np
from aeon.classification.convolution_based import MiniRocketClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.interval_based import (
    RandomIntervalSpectralEnsembleClassifier,
    TimeSeriesForestClassifier,
)
from aeon.datasets import load_classification
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CV = 5
RANDOM_STATE = 0

# Chosen by road-testing the pool: on these problems ensembling changes the
# result, and selection by accuracy and by log loss often disagree.
DATASETS = [
    "Computers",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW",
    "ECG200",
    "Herring",
    "InsectWingbeatSound",
    "Lightning2",
    "ToeSegmentation1",
]

DATA_DIR = Path(
    os.environ.get("COMP3222_UCR_DIR", Path(__file__).parent / "data" / "ucr")
)


def load_dataset(name):
    """Load the fixed train/test split of a univariate UCR problem.

    Parameters
    ----------
    name : str
        A name from DATASETS.

    Returns
    -------
    X_train, y_train, X_test, y_test : numpy.ndarray
        X arrays have shape (n_cases, series_length), one row per time series.
        y arrays hold the class labels as strings.
    """
    splits = []
    for split in ("train", "test"):
        X, y = load_classification(name, split=split, extract_path=DATA_DIR)
        if isinstance(X, list) or X.ndim != 3 or X.shape[1] != 1:
            raise ValueError(
                f"{name} is not an equal-length univariate problem."
            )
        splits += [X[:, 0, :], np.asarray(y)]
    return tuple(splits)


def make_pool():
    """Return a new list of (name, unfitted classifier) pairs.

    Each member uses a different representation of the series, and every
    member gives graded rather than one-hot probabilities. Ensemble sizes are
    reduced from the aeon defaults to keep the run time of Part 3 down.
    """
    return [
        # Raw series treated as tabular data, ignoring time order.
        ("rf", RandomForestClassifier(n_estimators=200,
                                      random_state=RANDOM_STATE)),
        ("logistic", make_pipeline(StandardScaler(),
                                   LogisticRegression(max_iter=1000))),
        ("naive_bayes", GaussianNB()),
        # Whole-series distance.
        ("knn_euclidean", KNeighborsTimeSeriesClassifier(
            distance="euclidean", n_neighbors=5)),
        # Summary statistics of random intervals.
        ("tsf", TimeSeriesForestClassifier(n_estimators=100,
                                           random_state=RANDOM_STATE)),
        # Spectral features of random intervals.
        ("rise", RandomIntervalSpectralEnsembleClassifier(
            n_estimators=50, random_state=RANDOM_STATE)),
        # 22 hand-crafted time series features.
        ("catch22", Catch22Classifier(random_state=RANDOM_STATE)),
        # Random convolutional kernels, with logistic regression in place of
        # the default ridge classifier so the probabilities are not one-hot.
        ("minirocket", MiniRocketClassifier(
            n_kernels=2000,
            estimator=make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=1000)),
            random_state=RANDOM_STATE)),
    ]


def make_stacking():
    """Return an unfitted StackingClassifier over a fresh copy of the pool."""
    return StackingClassifier(
        estimators=make_pool(),
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=CV, shuffle=True,
                           random_state=RANDOM_STATE),
    )
