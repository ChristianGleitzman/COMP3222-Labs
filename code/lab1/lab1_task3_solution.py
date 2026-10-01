"""Lab 1, Task 3: a binary classifier based on one feature."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class MedianThresholdRule(ClassifierMixin, BaseEstimator):
    """Predict each class on one side of the selected feature's median.

    The class with the lower mean feature value is predicted below the median.
    Values at the median are assigned to the other class.
    """

    def __init__(self, feature_index=0):
        self.feature_index = feature_index

    def fit(self, X, y):
        """Learn class order and the median threshold from training data."""
        X, y = check_X_y(X, y, dtype=float)
        if not 0 <= self.feature_index < X.shape[1]:
            raise ValueError("feature_index must identify a column of X")

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("MedianThresholdRule needs exactly two classes")

        column = X[:, self.feature_index]
        first, second = self.classes_
        first_mean = column[y == first].mean()
        second_mean = column[y == second].mean()
        self.less_class_ = first if first_mean <= second_mean else second
        self.greater_class_ = second if self.less_class_ == first else first
        self.threshold_ = np.median(column)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Apply the rule learned in fit to each row."""
        check_is_fitted(self, "threshold_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from the training data")
        return np.where(
            X[:, self.feature_index] < self.threshold_,
            self.less_class_,
            self.greater_class_,
        )


if __name__ == "__main__":
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    classifier = MedianThresholdRule().fit(X, y)
    print("Threshold:", classifier.threshold_)
    print("Predictions:", classifier.predict(X))
    print("Training accuracy:", classifier.score(X, y))
