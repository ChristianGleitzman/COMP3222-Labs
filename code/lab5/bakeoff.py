"""Lab 5, Task 2: comparing classifiers on multiple problems (student starter)."""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aeon.visualisation import plot_critical_difference, plot_pairwise_scatter
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from single_problem import make_classifiers
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "lab5"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "bakeoff.csv"
RANDOM_STATE = 42
MEASURES = ["accuracy", "balanced_accuracy", "fit_time"]

# UCI problems with continuous features, as in L5.2.
DATASETS = [
    "bank",
    "blood",
    "breast-cancer-wisc-diag",
    "breast-tissue",
    "cardiotocography-10clases",
    "conn-bench-sonar-mines-rocks",
    "conn-bench-vowel-deterding",
    "ecoli",
    "glass",
    "hill-valley",
    "image-segmentation",
    "ionosphere",
    "iris",
    "libras",
    "oocytes_merluccius_nucleus_4d",
    "oocytes_trisopterus_states_5b",
    "optical",
    "ozone",
    "page-blocks",
    "parkinsons",
    "planning",
    "post-operative",
    "ringnorm",
    "seeds",
    "spambase",
    "statlog-landsat",
    "statlog-vehicle",
    "steel-plates",
    "synthetic-control",
    "twonorm",
    "vertebral-column-3clases",
    "wall-following",
    "waveform-noise",
    "wine-quality-white",
    "yeast",
]


def load_dataset(name):
    """Return X (float array) and y (string labels) for one dataset.

    Each dataset is data/lab5/<name>/<name>.arff, with the class in the last column.
    """
    data, meta = arff.loadarff(DATA_DIR / name / f"{name}.arff")
    columns = meta.names()
    X = np.column_stack([data[c] for c in columns[:-1]]).astype(np.float64)
    y = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in data[columns[-1]]])
    return X, y


def run_one_dataset(name, classifiers, random_state=RANDOM_STATE):
    """Evaluate every classifier on one stratified 70/30 split of a dataset.

    Use the same train and test rows for every classifier. Clone each classifier
    before fitting, and time fit with time.perf_counter(). Return a list with one
    dict per classifier, with keys "dataset", "classifier" and those in MEASURES.
    """
    X, y = load_dataset(name)
    # Integer labels 0, 1, ... in sorted class order; BasicEnsemble's vote needs ints.
    y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)
    dataset_list = []
    for model_name, model in classifiers.items():
        curr_model = clone(model)
        start = time.perf_counter()
        curr_model.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        dataset_list.append({
            "dataset": name,
            "classifier": model_name,
            "accuracy": curr_model.score(X_test, y_test),
            "balanced_accuracy": balanced_accuracy_score(y_test, curr_model.predict(X_test)),
            "fit_time": fit_time
        })
    return dataset_list


def run_experiments(datasets=DATASETS, results_file=RESULTS_FILE):
    """Run each dataset in turn, appending its rows to results_file.

    Datasets already in the file are skipped, so you can stop and resume. Use a
    different results_file, or delete the old one, if you change the classifiers.
    """
    results_file.parent.mkdir(exist_ok=True)
    done = set(pd.read_csv(results_file)["dataset"]) if results_file.exists() else set()
    for name in datasets:
        if name in done:
            print(f"Skipping {name}: already in {results_file.name}")
            continue
        print(f"Running {name}")
        rows = run_one_dataset(name, make_classifiers(), RANDOM_STATE)
        pd.DataFrame(rows).to_csv(results_file, mode="a", index=False,
                                  header=not results_file.exists())
    return pd.read_csv(results_file)


def results_table(results, measure):
    """Return one row per dataset and one column per classifier for a measure."""
    table = results.pivot(index="dataset", columns="classifier", values=measure)
    order = [name for name in DATASETS if name in table.index]
    return table.loc[order, list(dict.fromkeys(results["classifier"]))]


def average_ranks(table, lower_better=False):
    """Return the average rank of each classifier (column) over the datasets.

    Rank 1 is best on a dataset. Tied classifiers share the average of their
    ranks; DataFrame.rank(axis=1, ...) does this.
    """
    # rank 1 is best, so rank ascending when lower is better and descending otherwise
    ranks = table.rank(axis=1, ascending=lower_better, method="average")
    return ranks.mean(axis=0)


def win_draw_loss(table, a, b, lower_better=False):
    """Return (wins, draws, losses) of classifier a against classifier b."""
    better = table[a] < table[b] if lower_better else table[a] > table[b]
    worse = table[a] > table[b] if lower_better else table[a] < table[b]
    wins, losses = int(better.sum()), int(worse.sum())
    return wins, len(table) - wins - losses, losses


def check_functions():
    """Compare your ranks and counts with a hand-worked table of two datasets."""
    table = pd.DataFrame({"A": [0.9, 0.7], "B": [0.8, 0.9], "C": [0.8, 0.6]},
                         index=["d1", "d2"])
    print("Average ranks, expected A 1.50, B 1.75, C 2.75")
    print(average_ranks(table).round(2).to_string())
    print("A v B expected (1, 0, 1):", win_draw_loss(table, "A", "B"))
    print("B v C expected (1, 1, 0):", win_draw_loss(table, "B", "C"))


if __name__ == "__main__":
    print(f"{len(DATASETS)} datasets in {DATA_DIR}")
    for name in DATASETS[:5]:
        X, y = load_dataset(name)
        print(f"  {name}: X {X.shape}, {len(np.unique(y))} classes")
    print(f"Classifiers: {', '.join(make_classifiers())}")

    # TODO: Implement run_one_dataset. Check it on five datasets first, with a
    #       separate file: run_experiments(DATASETS[:5], RESULTS_DIR / "check.csv").
    #run_experiments(DATASETS[:5], RESULTS_DIR / "check.csv")
    # TODO: Then run every dataset with run_experiments() and build
    #       results_table(results, measure) for each measure in MEASURES.
    #run_experiments()
    results = pd.read_csv(RESULTS_FILE)
    for m in MEASURES:
        print(results_table(results, m))
    check_functions()

    # Average ranks and pairwise win/draw/loss for each measure (lower fit time is better).
    names = list(make_classifiers())
    for m in MEASURES:
        table = results_table(results, m)
        lower = m == "fit_time"
        print(f"\n{m}: average ranks")
        print(average_ranks(table, lower_better=lower).round(2).to_string())
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                print(f"  {a} v {b} (win, draw, loss): {win_draw_loss(table, a, b, lower_better=lower)}")

    # Pairwise scatter of two classifiers, and critical difference diagrams for all three.
    acc = results_table(results, "accuracy")
    plot_pairwise_scatter(acc["RandF"].to_numpy(), acc["RotF"].to_numpy(), "RandF", "RotF",
                          metric="accuracy")
    plt.savefig(RESULTS_DIR / "scatter_RandF_RotF_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    for m in MEASURES:
        table = results_table(results, m)
        plot_critical_difference(table.to_numpy(), list(table.columns), test="nemenyi",
                                 alpha=0.05, lower_better=(m == "fit_time"))
        plt.savefig(RESULTS_DIR / f"cd_{m}.png", dpi=150, bbox_inches="tight")
        plt.close()
