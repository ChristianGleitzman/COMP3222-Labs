"""Lab 3, Task 3 solution: a simple majority-vote ensemble."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from lab3_task1 import load_credit_approval_data
from lab3_task1_solution import ImpactEncoderTransformer


class BasicEnsembleClassifier(ClassifierMixin, BaseEstimator):
    """Fit cloned classifiers on random subsets and combine their votes."""

    def __init__(self, base_estimator=None, n_estimators=10,
                 sample_fraction=0.6, replace=False, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.sample_fraction = sample_fraction
        self.replace = replace
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        base_estimator = self.base_estimator
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(max_depth=5)
        self.classes_ = np.unique(y)
        m = max(1, int(round(self.sample_fraction * len(X))))
        self.estimators_ = []
        for _ in range(self.n_estimators):
            selected = rng.choice(len(X), size=m, replace=self.replace)
            member = clone(base_estimator)
            # Trees break ties between equal splits at random, so seed each member too.
            if "random_state" in member.get_params():
                member.set_params(random_state=int(rng.integers(2**31)))
            member.fit(X[selected], y[selected])
            self.estimators_.append(member)
        return self

    def predict(self, X):
        predictions = np.array([member.predict(X) for member in self.estimators_])
        # votes[i, k] counts the members that predict class k for row i.
        votes = (predictions[:, :, None] == self.classes_).sum(axis=0)
        # argmax returns the first maximum, so ties go to the smallest label.
        return self.classes_[votes.argmax(axis=1)]

    def predict_proba(self, X):
        """Optional extension: average the members' probabilities."""
        proba = np.zeros((len(X), len(self.classes_)))
        for member in self.estimators_:
            # A member may not have seen every class in its sample.
            columns = np.searchsorted(self.classes_, member.classes_)
            proba[:, columns] += member.predict_proba(X)
        return proba / len(self.estimators_)


if __name__ == "__main__":
    from aeon.classification.sklearn import RotationForestClassifier

    X, y = load_credit_approval_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42
    )

    pipe = make_pipeline(ImpactEncoderTransformer(),
                         BasicEnsembleClassifier(random_state=42))
    pipe.fit(X_train, y_train)
    print(f"Basic ensemble pipeline test accuracy: "
          f"{accuracy_score(y_test, pipe.predict(X_test)):.3f}")

    # Optional extension: compare ensembles.
    for fraction in [0.3, 0.6, 1.0]:
        for replace in [False, True]:
            ensemble = BasicEnsembleClassifier(
                sample_fraction=fraction, replace=replace, random_state=42
            )
            pipe = make_pipeline(ImpactEncoderTransformer(), ensemble)
            pipe.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, pipe.predict(X_test))
            print(f"sample_fraction={fraction}, replace={replace}: {accuracy:.3f}")

    for clf in [RandomForestClassifier(random_state=42),
                RotationForestClassifier(random_state=42)]:
        pipe = make_pipeline(ImpactEncoderTransformer(), clf)
        pipe.fit(X_train, y_train)
        print(f"{type(clf).__name__}: {accuracy_score(y_test, pipe.predict(X_test)):.3f}")

    pipe = make_pipeline(ImpactEncoderTransformer(),
                         BasicEnsembleClassifier(random_state=42))
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)
    proba_accuracy = accuracy_score(y_test, pipe.classes_[proba.argmax(axis=1)])
    print(f"Basic ensemble, averaged probabilities: {proba_accuracy:.3f}")
