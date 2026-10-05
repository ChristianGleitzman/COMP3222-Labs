import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree, plot_tree


def load_data(split=None):
    """Load ItalyPowerDemand as a 2D feature matrix and integer class labels.

    Pass split="train" or "test" to load one partition; None loads both.
    aeon is needed only when this function is called.
    """
    from aeon.datasets import load_italy_power_demand

    X, y = load_italy_power_demand(split=split, return_type="numpy2d") # type: ignore
    return X, y.astype(np.int64)


def plot_tree_2d(
    clf,
    X,
    y=None,
    feature_names=None,
    class_names=None,
    ax=None,
    grid_points=400,
    margin=0.5,
    region_alpha=0.25,
    show_data=True,
    show_splits=True,
    annotate_depth=False,
    linewidth=2,
):
    """
    Plot decision regions and split lines for a fitted scikit-learn DecisionTree on 2D data.

    Parameters
    ----------
    clf : sklearn.tree.DecisionTreeClassifier or Regressor (fitted)
        Must expose .predict and .tree_.
    X : array-like of shape (n_samples, 2)
        Two input features.
    y : array-like of shape (n_samples,), optional
        Class labels for colouring markers. If None, points are unlabelled.
    feature_names : list/tuple of length 2, optional
        Axis labels.
    class_names : list-like of length n_classes, optional
        Legend labels for classes.
    ax : matplotlib Axes, optional
        Axes to draw on. If None, a new figure/axes is created.
    grid_points : int, default=400
        Resolution of the background decision map.
    margin : float, default=0.5
        Padding around min/max of each axis.
    region_alpha : float, default=0.25
        Transparency of the decision regions.
    show_data : bool, default=True
        Scatter the training points.
    show_splits : bool, default=True
        Draw split lines from the trained tree (solid for depth 0, dashed for 1,
        dotted for 2, dash-dot for 3+).
    annotate_depth : bool, default=False
        Add rough depth annotations on the plot.
    linewidth : float, default=2
        Line width for split lines.

    Returns
    -------
    ax : matplotlib Axes
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must be of shape (n_samples, 2).")

    # Create axes
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))

    # Bounds
    x0_min, x0_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    x1_min, x1_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    # Background decision regions
    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min, x0_max, grid_points),
        np.linspace(x1_min, x1_max, grid_points),
    )
    Z = clf.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)
    ax.contourf(xx0, xx1, Z, alpha=region_alpha)

    # Data points
    if show_data:
        if y is None:
            ax.scatter(X[:, 0], X[:, 1], marker="o", edgecolor="k")
        else:
            y = np.asarray(y)
            classes = np.unique(y)
            markers = ["o", "s", "^", "v", "P", "X", "D", "*", "<", ">", "h"]
            for i, cls in enumerate(classes):
                lab = (
                    class_names[int(cls)]
                    if (class_names is not None and int(cls) < len(class_names))
                    else str(cls)
                )
                ax.scatter(
                    X[y == cls, 0],
                    X[y == cls, 1],
                    marker=markers[i % len(markers)],
                    label=lab,
                )
            ax.legend(loc="best", framealpha=0.95)

    # Axis labels
    if feature_names is not None and len(feature_names) == 2:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])

    # Split lines from the trained tree
    if show_splits and hasattr(clf, "tree_"):
        tree = clf.tree_
        feat = tree.feature
        thr = tree.threshold
        left = tree.children_left
        right = tree.children_right
        styles = ["-", "--", ":", "-."]

        def draw(node, depth, xb, yb):
            f = feat[node]
            t = thr[node]
            if f == _tree.TREE_UNDEFINED:
                return
            style = styles[min(depth, 3)]
            if f == 0:  # vertical split (feature 0)
                x = float(t)
                x = min(max(x, xb[0]), xb[1])
                ax.plot([x, x], [yb[0], yb[1]], linestyle=style, linewidth=linewidth)
                draw(left[node], depth + 1, (xb[0], min(x, xb[1])), yb)
                draw(right[node], depth + 1, (max(x, xb[0]), xb[1]), yb)
            elif f == 1:  # horizontal split (feature 1)
                yline = float(t)
                yline = min(max(yline, yb[0]), yb[1])
                ax.plot([xb[0], xb[1]], [yline, yline], linestyle=style, linewidth=linewidth)
                draw(left[node], depth + 1, xb, (yb[0], min(yline, yb[1])))
                draw(right[node], depth + 1, xb, (max(yline, yb[0]), yb[1]))

        draw(0, 0, (x0_min, x0_max), (x1_min, x1_max))

        if annotate_depth:
            ax.text(x0_min + 0.35, x1_min + 0.3, "Depth=0")
            ax.text((x0_min + x0_max) / 2.6, (x1_min + x1_max) / 2.1, "Depth=1")
            ax.text(x0_max - 1.8, (x1_min + x1_max) / 2.3, "(Depth=2)")

    ax.set_xlim(x0_min, x0_max)
    ax.set_ylim(x1_min, x1_max)
    ax.set_title("Decision tree decision boundaries")
    plt.tight_layout()
    plt.show()
    return ax



def evaluate(X_tr, y_tr, X_te, y_te, **params):
    """Fit a tree with the given parameters and return its size and accuracies."""
    clf = DecisionTreeClassifier(random_state=0, **params).fit(X_tr, y_tr)
    return {
        "depth": clf.get_depth(),
        "leaves": clf.get_n_leaves(),
        "train_acc": clf.score(X_tr, y_tr),
        "test_acc": clf.score(X_te, y_te),
    }


def print_table(title, rows):
    """Print one row per setting: label, tree size, train and test accuracy."""
    print(f"\n{title}")
    print(f"{'setting':<22}{'depth':>6}{'leaves':>8}{'train':>8}{'test':>8}")
    for label, r in rows:
        print(
            f"{label:<22}{r['depth']:>6}{r['leaves']:>8}"
            f"{r['train_acc']:>8.3f}{r['test_acc']:>8.3f}"
        )


def sweep(X_tr, y_tr, X_te, y_te, param, values):
    """Evaluate one regularisation parameter over a range of values."""
    return [(v, evaluate(X_tr, y_tr, X_te, y_te, **{param: v})) for v in values]


def plot_sweeps(sweeps):
    """One panel per parameter: train/test accuracy (left axis), leaves (right axis)."""
    fig, axes = plt.subplots(1, len(sweeps), figsize=(4.5 * len(sweeps), 4))
    for ax, (param, results) in zip(np.atleast_1d(axes), sweeps.items()):
        labels = [str(v) for v, _ in results]
        pos = np.arange(len(results))
        ax.plot(pos, [r["train_acc"] for _, r in results], "o-", label="train acc")
        ax.plot(pos, [r["test_acc"] for _, r in results], "s-", label="test acc")
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_xlabel(param)
        ax.set_ylabel("accuracy")
        ax.set_ylim(0.8, 1.02)
        ax2 = ax.twinx()
        ax2.bar(pos, [r["leaves"] for _, r in results], alpha=0.15, color="grey")
        ax2.set_ylabel("leaves (grey bars)")
        ax.set_title(f"Effect of {param}")
        ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_trees_side_by_side(X_tr, y_tr, settings):
    """Draw fitted trees for a few settings so the change in shape is visible."""
    fig, axes = plt.subplots(1, len(settings), figsize=(6 * len(settings), 5))
    for ax, (label, params) in zip(np.atleast_1d(axes), settings.items()):
        clf = DecisionTreeClassifier(random_state=0, **params).fit(X_tr, y_tr)
        plot_tree(clf, ax=ax, filled=True, impurity=False, label="none")
        ax.set_title(f"{label}\ndepth={clf.get_depth()}, leaves={clf.get_n_leaves()}")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Task 3: how each DecisionTreeClassifier parameter changes the tree.
    # The official train split has only 67 series (a depth-3 tree fits it perfectly),
    # so pool both partitions and make a stratified 50/50 split to get trees big enough
    # for the settings to matter. Fit on the train half, score on the held-out half.
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=0
    )
    print("Train:", X_tr.shape, "Test:", X_te.shape, "Classes:", np.unique(y_tr))

    # 1. criterion and splitter (full-depth trees, so these only change which splits are chosen)
    rows = []
    for criterion in ["gini", "entropy", "log_loss"]:
        for splitter in ["best", "random"]:
            rows.append(
                (f"{criterion}/{splitter}",
                 evaluate(X_tr, y_tr, X_te, y_te, criterion=criterion, splitter=splitter))
            )
    print_table("criterion / splitter (no regularisation)", rows)

    # 2. regularisation: sweep one parameter at a time, others left at default
    sweeps = {
        "max_depth": sweep(X_tr, y_tr, X_te, y_te, "max_depth", [1, 2, 3, 4, 6, 8, 12, None]),
        "min_samples_leaf": sweep(X_tr, y_tr, X_te, y_te, "min_samples_leaf", [1, 2, 5, 10, 20, 50, 100]),
        "max_leaf_nodes": sweep(X_tr, y_tr, X_te, y_te, "max_leaf_nodes", [2, 4, 8, 16, 32, 64, None]),
        "ccp_alpha": sweep(X_tr, y_tr, X_te, y_te, "ccp_alpha", [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]),
    }
    for param, results in sweeps.items():
        print_table(param, [(str(v), r) for v, r in results])

    # Best setting per parameter by test accuracy (illustrative; use CV to pick for real)
    print("\nBest value per parameter by test accuracy:")
    for param, results in sweeps.items():
        v, r = max(results, key=lambda vr: vr[1]["test_acc"])
        print(f"  {param}={v}: test acc {r['test_acc']:.3f} with {r['leaves']} leaves")

    plot_sweeps(sweeps)
    plot_trees_side_by_side(
        X_tr, y_tr,
        {
            "unrestricted": {},
            "max_depth=3": {"max_depth": 3},
            "ccp_alpha=0.005": {"ccp_alpha": 0.005},
        },
    )
    plt.show()
