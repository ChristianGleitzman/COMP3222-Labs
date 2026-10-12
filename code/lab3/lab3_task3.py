"""Lab 3, Task 3: a simple majority-vote ensemble (student starter)."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from lab3_task1 import ImpactEncoderTransformer, load_credit_approval_data


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
        rng = np.random.default_rng(self.random_state)
        base = self.base_estimator
        if base is None:
            base = DecisionTreeClassifier(max_depth=5)

        # For each member, clone the base estimator, draw row indices,
        # fit it on the sampled rows and store it in self.estimators_.
        ensemble = []
        for i in range(self.n_estimators):
            m = max(1, int(round(self.sample_fraction * len(X))))
            selected = rng.choice(len(X), size=m, replace=self.replace)
            X_sample, y_sample = X[selected], y[selected]
            member = clone(base)
            member.fit(X_sample, y_sample)
            ensemble.append(member)

        self.estimators_ = ensemble
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        # Collect each member's predictions, then vote row by row.
        preds = []
        for e in self.estimators_:
            preds.append(e.predict(X))
        preds = np.array(preds)

        votes = []
        for j in range(preds.shape[1]):
            curr = preds[:, j]
            vote = np.argmax(np.bincount(curr))
            votes.append(vote)
        return np.array(votes)

    # Optional extension: add predict_proba by averaging member probabilities.

    def predict_proba(self, X):
        preds = []
        for e in self.estimators_:
            preds.append(e.predict_proba(X))
        preds = np.array(preds)  # shape (n_estimators, n_rows, n_classes)

        # Average over the estimators axis and return an array of shape
        # (n_rows, n_classes) whose rows sum to 1.
        averages = preds.mean(axis=0)
        return averages


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    X, y = load_credit_approval_data()
    print(f"Credit approval: {X.shape[0]} complete rows, {X.shape[1]} features")
    print(f"Ensemble defaults: {BasicEnsembleClassifier().get_params()}")

    # Split first so the encoder only ever sees training rows.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    pipe = Pipeline([
        ("encode", ImpactEncoderTransformer()),
        ("cls", BasicEnsembleClassifier(random_state=0)),
    ])
    pipe.fit(X_train, y_train)
    print(f"Pipeline test accuracy: {pipe.score(X_test, y_test):.3f}")

    # Optional extension: compare ensembles on the same 70/30 split.
    from aeon.classification.sklearn import RotationForestClassifier
    from sklearn.ensemble import RandomForestClassifier

    # Preprocessing is fitted on training rows only, then reused for the test rows.
    encoder = ImpactEncoderTransformer().fit(X_train, y_train)
    Xtr, Xte = encoder.transform(X_train), encoder.transform(X_test)

    print("\nTest accuracy by model")
    models = {
        "BasicEnsemble": BasicEnsembleClassifier(random_state=0),
        "RandomForest": RandomForestClassifier(random_state=0),
        "RotationForest": RotationForestClassifier(random_state=0),
    }
    for name, model in models.items():
        model.fit(Xtr, y_train)
        print(f"  {name:15s} {model.score(Xte, y_test):.3f}")

    # How do sample_fraction and replace change the ensemble? A single split is
    # noisy, so average over several seeds (each with a different member sampling).
    print("\nBasicEnsemble: mean test accuracy over 5 seeds")
    for replace in (False, True):
        for fraction in (0.3, 0.6, 0.9):
            scores = [
                BasicEnsembleClassifier(
                    sample_fraction=fraction, replace=replace, random_state=seed
                ).fit(Xtr, y_train).score(Xte, y_test)
                for seed in range(5)
            ]
            print(f"  replace={replace!s:5s} fraction={fraction}: {np.mean(scores):.3f}")
