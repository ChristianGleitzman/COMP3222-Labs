"""Lab 3, Task 3: a simple majority-vote ensemble (student starter)."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier


class BasicEnsembleClassifier(ClassifierMixin, BaseEstimator):
    """Fit cloned classifiers on random subsets and combine their votes.

    If base_estimator is None, use DecisionTreeClassifier(max_depth=5).
    With replace=False, each member samples without replacement; True gives
    bootstrap samples. A tied vote should choose the smallest class label.
    """

    def __init__(self, base_estimator=None, n_estimators=10,
                 sample_fraction=0.6, replace=False, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.sample_fraction = sample_fraction
        self.replace = replace
        self.random_state = random_state

    def fit(self, X, y):
        # TODO: Use np.random.default_rng(self.random_state).
        # TODO: Choose the default tree if base_estimator is None.
        # TODO: For each member, clone the base estimator, draw row indices,
        # fit it on the sampled rows and store it in self.estimators_.
        # m = max(1, int(round(self.sample_fraction * len(X))))
        # selected = rng.choice(len(X), size=m, replace=self.replace)
        # member = clone(base_estimator)
        raise NotImplementedError("Implement BasicEnsembleClassifier.fit")

    def predict(self, X):
        # TODO: Collect each member's predictions, then vote row by row.
        raise NotImplementedError("Implement BasicEnsembleClassifier.predict")

    # Optional extension: add predict_proba by averaging member probabilities.


if __name__ == "__main__":
    from lab3_task1 import load_credit_approval_data

    X, y = load_credit_approval_data()
    print(f"Credit approval: {X.shape[0]} complete rows, {X.shape[1]} features")
    print(f"Ensemble defaults: {BasicEnsembleClassifier().get_params()}")
    print("Implement fit and predict, then test them in a pipeline.")
