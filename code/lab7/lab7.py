"""Lab 7: neural networks for classification and regression (student starter)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DOLLARS_PER_UNIT = 100_000  # the housing target is in units of $100,000

# For B3 and C3. Each parameter must be prefixed with the name of its step in the
# pipeline, for example "mlpclassifier__alpha" (make_pipeline lower-cases the class name).
CLASSIFIER_GRID = {
    "alpha": [1e-4, 1e-2, 1],
    "hidden_layer_sizes": [(16,), (64,), (64, 32)],
}
REGRESSOR_GRID = {
    "alpha": [1e-4, 1e-2],
    "learning_rate_init": [1e-3, 1e-2],
}


def show(save_as=None):
    """Save the current figure (if asked) and then show it."""
    if save_as:
        RESULTS_DIR.mkdir(exist_ok=True)
        plt.savefig(RESULTS_DIR / save_as, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# Part A: breast cancer diagnosis.

def load_classification_data():
    """Return X_train, X_test, y_train, y_test for a stratified 80/20 split.

    The labels are NOT recoded here: 0 = malignant and 1 = benign, as in the sheet.
    """
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def evaluate_classifier(model, X_test, y_test):
    """Return a dict with the test "accuracy", "roc_auc" and "confusion_matrix"."""

    return {
        "accuracy": model.score(X_test, y_test),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        "confusion_matrix": confusion_matrix(y_test, model.predict(X_test))
    }


def print_confusion(cm, names=("malignant", "benign")):
    """Print a confusion matrix with actual classes as rows, in label order 0, 1."""
    print(pd.DataFrame(cm,
                       index=[f"actual {n}" for n in names],
                       columns=[f"predicted {n}" for n in names]))


def threshold_errors(y_true, proba_benign, threshold):
    """Predict benign (1) when P(benign) > threshold. Return (missed, false_alarms).

    A missed malignant tumour is a true 0 predicted as 1. A false alarm is a true 1
    predicted as 0.
    """
    predicted = (np.asarray(proba_benign) > threshold).astype(int)
    y_true = np.asarray(y_true)
    missed = int(np.sum((y_true == 0) & (predicted == 1)))
    false_alarms = int(np.sum((y_true == 1) & (predicted == 0)))
    return missed, false_alarms


# Part B: California house prices.

def load_regression_data():
    """Return X_train, X_test, y_train, y_test for an 80/20 split.

    The target is the median house value in units of $100,000. The first call
    downloads the dataset, so it needs an internet connection.
    """
    X, y = fetch_california_housing(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def evaluate_regressor(model, X, y):
    """Return a dict with "rmse" and "mae" in dollars, and "r2".

    Use it on the test set, and on the training set for the training RMSE in C2.
    """
    y_pred = model.predict(X)
    return {
        "rmse": np.sqrt(mean_squared_error(y, y_pred)) * DOLLARS_PER_UNIT,
        "mae": mean_absolute_error(y, y_pred) * DOLLARS_PER_UNIT,
        "r2": r2_score(y, y_pred),
    }


def plot_training_curves(mlp, save_as=None):
    """Plot loss_curve_ and validation_scores_ against epoch for a fitted MLPRegressor."""
    fig, (ax_loss, ax_val) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(range(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training loss")
    ax_loss.set_title("Training loss")
    ax_val.plot(range(1, len(mlp.validation_scores_) + 1), mlp.validation_scores_)
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Validation score (R$^2$)")
    ax_val.set_title("Validation score")
    fig.tight_layout()
    show(save_as)


def plot_predicted_vs_actual(y_true, y_pred, save_as=None):
    """Plot predicted against actual values, with the line y = x."""
    plt.scatter(y_true, y_pred, s=5, alpha=0.3)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    plt.plot(lims, lims, "r--", label="y = x")
    plt.xlabel("Actual value ($100,000)")
    plt.ylabel("Predicted value ($100,000)")
    plt.title("Predicted against actual (test set)")
    plt.legend()
    show(save_as)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_classification_data()
    print(f"Breast cancer: {X_train.shape[0]} train, {X_test.shape[0]} test, "
          f"{X_train.shape[1]} features")
    print(f"  malignant (0) in train: {np.sum(y_train == 0)}, benign (1): {np.sum(y_train == 1)}")

    # TODO B1: Baseline. Train a LogisticRegression pipeline. Report test accuracy,
    #       ROC-AUC and the confusion matrix with evaluate_classifier.
    # TODO B2: Train an MLPClassifier with hidden_layer_sizes=(32,) in a pipeline and
    #       report the same three results. Is the MLP worth its extra complexity?
    # TODO B3: Use GridSearchCV with 5-fold cross-validation (scoring="roc_auc") on the
    #       training set only, over CLASSIFIER_GRID. Report the best settings and their
    #       test ROC-AUC.
    # TODO B4: From your best model's confusion matrix, how many malignant tumours were
    #       missed? Use predict_proba and threshold_errors to raise the threshold on
    #       P(benign) to 0.8 and then 0.9, and report the missed malignant tumours and
    #       false alarms at each. Which threshold would you recommend, and why?

    regression_model = make_pipeline(StandardScaler(), LogisticRegression())
    regression_model.fit(X_train, y_train)
    print(evaluate_classifier(regression_model, X_test, y_test))

    mlp_model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,)))
    mlp_model.fit(X_train, y_train)
    print(evaluate_classifier(mlp_model, X_test, y_test))

    # B3: grid search on the training set only. The scaler is inside the pipeline, so
    # it is refitted on each training fold. Grid keys need the "mlpclassifier__" prefix.
    mlp_pipeline = make_pipeline(StandardScaler(),
                                 MLPClassifier(max_iter=1000, random_state=RANDOM_STATE))
    param_grid = {f"mlpclassifier__{name}": values for name, values in CLASSIFIER_GRID.items()}
    search = GridSearchCV(mlp_pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"\nB3 best settings: {search.best_params_}")
    print(f"B3 best cross-validated ROC-AUC: {search.best_score_:.4f}")
    best_model = search.best_estimator_  # refitted on all the training data
    best_results = evaluate_classifier(best_model, X_test, y_test)
    print(f"B3 test accuracy {best_results['accuracy']:.4f}, "
          f"test ROC-AUC {best_results['roc_auc']:.4f}")
    print_confusion(best_results["confusion_matrix"])

    # B4: raise the threshold on P(benign) and count the two kinds of mistake.
    proba_benign = best_model.predict_proba(X_test)[:, 1]
    print("\nB4 threshold   missed malignant   false alarms")
    for threshold in (0.5, 0.8, 0.9):
        missed, false_alarms = threshold_errors(y_test, proba_benign, threshold)
        print(f"{threshold:>12.1f}   {missed:>16d}   {false_alarms:>12d}")

    X_train_h, X_test_h, y_train_h, y_test_h = load_regression_data()
    print(f"\nCalifornia housing: {X_train_h.shape[0]} train, {X_test_h.shape[0]} test, "
          f"{X_train_h.shape[1]} features")
    print(f"  target range: {y_train_h.min():.2f} to {y_train_h.max():.2f} "
          f"(units of ${DOLLARS_PER_UNIT:,})")

    def print_regression(name, results):
        print(f"{name}: RMSE ${results['rmse']:,.0f}, MAE ${results['mae']:,.0f}, "
              f"R2 {results['r2']:.4f}")

    # C1: linear regression baseline.
    linear = make_pipeline(StandardScaler(), LinearRegression())
    linear.fit(X_train_h, y_train_h)
    print_regression("\nC1 linear regression (test)", evaluate_regressor(linear, X_test_h, y_test_h))

    # C2: MLP regressor with early stopping, so validation_scores_ is recorded.
    mlp_reg = make_pipeline(StandardScaler(),
                            MLPRegressor(hidden_layer_sizes=(64, 64), early_stopping=True,
                                         max_iter=500, random_state=RANDOM_STATE))
    mlp_reg.fit(X_train_h, y_train_h)
    print_regression("C2 MLP (test)", evaluate_regressor(mlp_reg, X_test_h, y_test_h))
    print_regression("C2 MLP (train)", evaluate_regressor(mlp_reg, X_train_h, y_train_h))
    plot_training_curves(mlp_reg[-1], save_as="c2_training_curves.png")

    # C3: small grid search, 3-fold, scored by RMSE.
    reg_grid = {f"mlpregressor__{name}": values for name, values in REGRESSOR_GRID.items()}
    reg_search = GridSearchCV(
        make_pipeline(StandardScaler(),
                      MLPRegressor(hidden_layer_sizes=(64, 64), early_stopping=True,
                                   max_iter=500, random_state=RANDOM_STATE)),
        reg_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
    reg_search.fit(X_train_h, y_train_h)
    print(f"\nC3 best settings: {reg_search.best_params_}")
    print(f"C3 best cross-validated RMSE: ${-reg_search.best_score_ * DOLLARS_PER_UNIT:,.0f}")
    print_regression("C3 best model (test)",
                     evaluate_regressor(reg_search.best_estimator_, X_test_h, y_test_h))

    # C4: predicted against actual for the best model on the test set.
    plot_predicted_vs_actual(y_test_h, reg_search.best_estimator_.predict(X_test_h),
                             save_as="c4_predicted_vs_actual.png")


    # TODO C1: Baseline. Train a LinearRegression pipeline. Report test RMSE, MAE and
    #       R2, with RMSE and MAE in dollars.
    # TODO C2: Train an MLPRegressor with hidden_layer_sizes=(64, 64) in a pipeline.
    #       Report the same metrics and the training RMSE. Plot loss_curve_ and
    #       validation_scores_ against epoch with plot_training_curves
    #       (validation_scores_ is only recorded when early_stopping=True).
    #       Is there a large gap between training and test error? What does it tell you?
    # TODO C3: Use GridSearchCV (3-fold, scoring="neg_root_mean_squared_error") over
    #       REGRESSOR_GRID. Report the best settings and their test RMSE.
    # TODO C4: Plot predicted against actual values for the test set with
    #       plot_predicted_vs_actual. Where are the largest errors? How does the
    #       $500,000 cap on the highest-priced houses affect the model?
