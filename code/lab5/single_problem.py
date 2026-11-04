"""Lab 5, Task 1: comparing classifiers on a single problem (student starter)."""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aeon.classification.sklearn import RotationForestClassifier
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
# "code" clashes with Python's built-in code module, and the Lab 3 files use flat
# imports, so add code/lab3 to the path and import from it directly.
sys.path.append(str(Path(__file__).resolve().parents[1] / "lab3"))
from lab3_task3 import BasicEnsembleClassifier  # type: ignore

RANDOM_STATE = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_breast_cancer_data():
    """Return X, y and class names, with malignant as the positive class 1.

    The scikit-learn loader codes malignant as 0 and benign as 1, so the labels
    are recoded here. class_names[label] gives the name of each label.
    """
    data = load_breast_cancer()
    y = 1 - data.target
    return data.data, y, ["benign", "malignant"]


def load_iris_data():
    """Return X, y and class names for Iris (labels 0, 1 and 2)."""
    data = load_iris()
    return data.data, data.target, list(data.target_names)


def make_classifiers(random_state=RANDOM_STATE):
    """Return the three classifiers to compare, keyed by a short name.

    Put preprocessing and your classifier in a Pipeline if it needs any, so the
    preprocessing is fitted on training data only. bakeoff.py reuses this function.
    """
    return {
        "RandF": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "RotF": RotationForestClassifier(n_estimators=100, random_state=random_state),
        "BasicEnsemble": BasicEnsembleClassifier(n_estimators=100, random_state=random_state),
    }


def single_split(y, random_state=RANDOM_STATE):
    """Return train and test row indices for a stratified 70/30 split.

    The row indices are the case identifiers to save with your predictions.
    """
    ids = np.arange(len(y))
    return train_test_split(ids, test_size=0.3, stratify=y, random_state=random_state)


def make_folds(random_state=RANDOM_STATE):
    """The 5-fold stratified cross-validation used in 1.2 and the nested extension."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)


def class_column(model, label):
    """Return the predict_proba column of label, using the order in model.classes_."""
    return int(np.flatnonzero(model.classes_ == label)[0])


def labelled_confusion_matrix(y_true, y_pred, class_names):
    """Return the confusion matrix as a DataFrame with actual classes as rows."""
    labels = np.arange(len(class_names))
    counts = pd.crosstab(pd.Categorical(y_true, categories=labels),
                         pd.Categorical(y_pred, categories=labels), dropna=False)
    counts.index = [f"actual {name}" for name in class_names]
    counts.columns = [f"predicted {name}" for name in class_names]
    return counts


def save_predictions(path, ids, y_true, y_pred, proba, class_names):
    """Save case ids, true labels, predictions and class probabilities to a CSV."""
    table = pd.DataFrame({"case_id": ids, "true": y_true, "predicted": y_pred})
    for k, name in enumerate(class_names):
        table[f"p_{name}"] = proba[:, k]
    table.to_csv(path, index=False)


def compute_metrics(y_true, y_pred, proba):
    """Return a dict of the 1.1 measures for one classifier.

    proba has one column per class, in label order 0, 1, ... For both problems
    return accuracy, balanced accuracy and log loss. With two classes, also return
    precision, recall, F1 and AUROC for the positive class 1. With more classes,
    return macro-averaged precision, recall and F1 and macro one-versus-rest AUROC.
    See L5.1 slides 31 and 32.
    """
    n_classes = proba.shape[1]
    metrics = {}

    # TODO: measures shared by both problems.
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["log_loss"] = log_loss(y_true, proba) # Log loss relies on confidence in predictions

    if n_classes == 2:
        # TODO: positive class is 1. AUROC needs the class-1 column, proba[:, 1].
        metrics["precision"] = precision_score(y_true, y_pred)
        metrics["recall"] = recall_score(y_true, y_pred)
        metrics["f1"] = f1_score(y_true, y_pred)
        metrics["auroc"] = roc_auc_score(y_true, proba[:,1])
    else:
        # macro-averaged precision, recall and F1. macro is the plain unweighted mean of the scores on each class
        # macro one-versus-rest AUROC, which uses the full proba matrix.
        metrics["precision"] = precision_score(y_true, y_pred, average="macro")
        metrics["recall"] = recall_score(y_true, y_pred, average="macro")
        metrics["f1"] = f1_score(y_true, y_pred, average="macro")
        metrics["auroc"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")

    return metrics


def check_metrics():
    """Compare compute_metrics with hand-worked values for four cases."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    proba = np.array([[0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.9, 0.1]])
    expected = {"accuracy": 0.75, "balanced_accuracy": 0.75, "precision": 1.0,
                "recall": 0.5, "f1": 0.667, "auroc": 1.0, "log_loss": 0.400}
    found = compute_metrics(y_true, y_pred, proba)
    for name, value in expected.items():
        print(f"{name:18s} expected {value:.3f}  yours {found.get(name, float('nan')):.3f}")


def cross_val_predictions(model, X, y, folds):
    """Return out-of-fold predictions and probabilities for every case.

    For each fold, clone the model, fit it on the training rows and predict the
    held-out rows, so no case is predicted by a model that saw it. Store the
    probabilities in label order, using class_column to find each column.
    """
    classes = np.unique(y)
    y_pred = np.empty(len(y), dtype=y.dtype)
    proba = np.zeros((len(y), len(classes)))
    for train_idx, test_idx in folds.split(X, y):
        fold_model = clone(model).fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = fold_model.predict(X[test_idx])
        fold_proba = fold_model.predict_proba(X[test_idx])
        for k, label in enumerate(classes):
            proba[test_idx, k] = fold_proba[:, class_column(fold_model, label)]
    return y_pred, proba


# For 1.3. Keys must match make_classifiers. If a classifier is in a Pipeline,
# prefix each parameter with the step name, for example "randomforestclassifier__max_depth".
PARAM_GRIDS = {
    "RandF": {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 5]},
    "RotF": {"max_group": [3, 5, 7], "remove_proportion": [0.25, 0.5]},
    "BasicEnsemble": {"sample_fraction": [0.4, 0.6, 0.8], "replace": [False, True]}
}

def tune(model, param_grid, X_train, y_train, random_state=RANDOM_STATE):
    """Return a fitted GridSearchCV that selects parameters by accuracy.

    Use 5-fold stratified cross-validation on the training data only. With the
    default refit=True, the best configuration is refitted on all training data.
    """
    # TODO: build the search (the GridSearchCV import is already at the top),
    # fit it on the training data and return it.
    cv = make_folds(random_state)
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        refit=True,
        verbose=1
    )
    return search.fit(X_train, y_train)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    problems = {"breast_cancer": load_breast_cancer_data(), "iris": load_iris_data()}
    for name, (X, y, class_names) in problems.items():
        counts = ", ".join(f"{c} {n}" for c, n in zip(class_names, np.bincount(y)))
        train_ids, test_ids = single_split(y)
        print(f"{name}: X {X.shape}, classes: {counts}")
        print(f"  split seed {RANDOM_STATE}: {len(train_ids)} train, {len(test_ids)} test")
        y_true = y[test_ids]
        results = {}
        for cls_name, cls in make_classifiers().items():
            cls.fit(X[train_ids], y[train_ids])
            print(f"  {cls_name} classes_: {cls.classes_}")
            y_pred = cls.predict(X[test_ids])
            proba = cls.predict_proba(X[test_ids])
            results[cls_name] = compute_metrics(y_true, y_pred, proba)
            print(labelled_confusion_matrix(y_true, y_pred, class_names), "\n")
            save_predictions(RESULTS_DIR / f"{name}_{cls_name}_predictions.csv",
                             test_ids, y_true, y_pred, proba, class_names)
        table = pd.DataFrame(results).T
        print(table.round(4), "\n")
        table.to_csv(RESULTS_DIR / f"{name}_metrics.csv")

    print(f"Classifiers: {', '.join(make_classifiers())}")
    print(f"Results saved in {RESULTS_DIR}")

    # 1.1: Implement compute_metrics, then call check_metrics().
    check_metrics()
    # TODO 1.1: For each problem, fit every classifier on the same training rows.
    #       Report the measures and a labelled confusion matrix on the test rows.
    #       Save a results CSV, and the predictions with save_predictions.
    # 1.2: 5-fold cross-validation, same folds for every classifier, metrics from
    # the pooled out-of-fold predictions.
    for name, (X, y, class_names) in problems.items():
        print(f"{name}: 5-fold cross-validation")
        folds = make_folds()
        cv_results = {}
        for cls_name, cls in make_classifiers().items():
            y_pred, proba = cross_val_predictions(cls, X, y, folds)
            cv_results[cls_name] = compute_metrics(y, y_pred, proba)
            if name == "breast_cancer":
                fpr, tpr, _ = roc_curve(y, proba[:, 1])
                plt.plot(fpr, tpr, label=f"{cls_name} (AUROC {cv_results[cls_name]['auroc']:.3f})")
        cv_table = pd.DataFrame(cv_results).T
        print(cv_table.round(4), "\n")
        cv_table.to_csv(RESULTS_DIR / f"{name}_cv_metrics.csv")
        if name == "breast_cancer":
            plt.plot([0, 1], [0, 1], "k--", label="chance")
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate (malignant)")
            plt.title("Breast cancer ROC, pooled 5-fold out-of-fold predictions")
            plt.legend()
            plt.savefig(RESULTS_DIR / "breast_cancer_roc.png", dpi=150, bbox_inches="tight")
            plt.close()
    # 1.3: Tune every classifier that has an entry in PARAM_GRIDS on the 1.1
    # training rows, then compare with the untuned 1.1 test results.
    # TODO: add a second grid to PARAM_GRIDS above; this loop picks it up.
    for name, (X, y, class_names) in problems.items():
        print(f"{name}: tuning")
        train_ids, test_ids = single_split(y)
        untuned = pd.read_csv(RESULTS_DIR / f"{name}_metrics.csv", index_col=0)
        tuned_results = {}
        for cls_name, grid in PARAM_GRIDS.items():
            start = time.perf_counter()
            search = tune(make_classifiers()[cls_name], grid, X[train_ids], y[train_ids])
            fit_time = time.perf_counter() - start
            print(f"  {cls_name}: best_params_ {search.best_params_}, "
                  f"CV accuracy {search.best_score_:.4f}, tuning time {fit_time:.1f}s")
            y_pred = search.predict(X[test_ids])
            proba = search.predict_proba(X[test_ids])
            tuned_results[cls_name] = compute_metrics(y[test_ids], y_pred, proba)
            print(f"  {cls_name}: test accuracy untuned {untuned.loc[cls_name, 'accuracy']:.4f}, "
                  f"tuned {tuned_results[cls_name]['accuracy']:.4f}")
        pd.DataFrame(tuned_results).T.to_csv(RESULTS_DIR / f"{name}_tuned_metrics.csv")
