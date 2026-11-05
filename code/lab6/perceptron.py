"""Lab 6: the perceptron (student starter)."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Part 1 data: logical AND.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
t = np.array([0, 0, 0, 1])  # AND targets

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def plot_boundary(predict, X, t, title="", save_as=None):
    """Shade the regions that any classifier assigns to each class. Reused throughout.

    If save_as is given, the figure is also saved to RESULTS_DIR / save_as.
    """
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300))
    zz = predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], colors=["#ffcccc", "#cce0ff"])
    plt.scatter(X[t == 0, 0], X[t == 0, 1], c="red", marker="o", label="Class 0")
    plt.scatter(X[t == 1, 0], X[t == 1, 1], c="blue", marker="x", s=80, label="Class 1")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(title)
    plt.legend()
    show(save_as)


def show(save_as=None):
    """Save the current figure (if asked) and then show it."""
    if save_as:
        RESULTS_DIR.mkdir(exist_ok=True)
        plt.savefig(RESULTS_DIR / save_as, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_history(history, title, save_as=None):
    """Plot the mistakes made in each epoch."""
    plt.plot(range(1, len(history) + 1), history, marker="o" if len(history) < 40 else None)
    plt.xlabel("Epoch")
    plt.ylabel("Mistakes")
    plt.title(title)
    show(save_as)


# Part 1: teaching a perceptron logical AND.

def add_bias(X):
    """Add the x0 = 1 column."""
    return np.hstack([np.ones((len(X), 1)), X])


def perceptron_fit(X, t, eta=1.0, w_init=None, max_epochs=100):
    """Train a perceptron and return the weights and the mistakes in each epoch."""
    Xb = add_bias(X)
    w = np.zeros(Xb.shape[1]) if w_init is None else np.array(w_init, dtype=float)
    mistakes_per_epoch = []
    for epoch in range(max_epochs):
        mistakes = 0
        for x, target in zip(Xb, t):
            y = 1 if w @ x > 0 else 0  # TODO: 1 if w @ x > 0, else 0
            if y != target:
                w = w - eta * (y - target) * x
                mistakes += 1
        mistakes_per_epoch.append(mistakes)
        if mistakes == 0:  # TODO: stop when converged
            break
    return w, mistakes_per_epoch


def perceptron_predict(w, X):
    return (add_bias(X) @ w > 0).astype(int)


# Part 2 data: two Gaussian classes, 30% kept aside for validation.

def make_gaussian_data():
    """Return X_tr, X_va, t_tr, t_va for two classes of 100 points each."""
    rng = np.random.default_rng(0)  # use this seed
    C = np.array([[2, 1], [1, 2]])  # shared covariance
    data0 = rng.multivariate_normal(mean=[0, 3], cov=C, size=100)  # class 0
    data1 = rng.multivariate_normal(mean=[3, 0], cov=C, size=100)  # class 1
    X2 = np.vstack([data0, data1])
    t2 = np.r_[np.zeros(100, dtype=int), np.ones(100, dtype=int)]
    return train_test_split(X2, t2, test_size=0.3, stratify=t2, random_state=0)


# Part 3: different activation functions, trained by gradient descent.

def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # TODO: 1 / (1 + e^(-z))


def tanh(z):
    return np.tanh(z)  # TODO: use np.tanh


def relu(z):
    return np.maximum(0,z)  # TODO: max(0, z), use np.maximum


ACTIVATIONS = {  # name: (f, f'(a, z), targets for class 0/1, threshold)
    "sigmoid": (sigmoid, lambda a, z: a * (1 - a), (0, 1), 0.5),
    "tanh": (tanh, lambda a, z: 1 - a**2, (-1, 1), 0.0),
    "relu": (relu, lambda a, z: (z > 0).astype(float), (0, 1), 0.5),
}


def neuron_fit(X, t, act, eta=0.1, epochs=3000):
    f, df, (lo, hi), _ = ACTIVATIONS[act]
    target = np.where(t == 1, hi, lo)  # targets in this activation's range
    Xb = add_bias(X)
    w = np.random.default_rng(0).normal(0, 0.1, Xb.shape[1])  # small random start
    for epoch in range(epochs):
        z = Xb @ w
        a = f(z)
        w = w - eta * Xb.T @ ((a - target) * df(a, z)) / len(t)
    return w


def neuron_predict(w, X, act):
    f, _, _, threshold = ACTIVATIONS[act]
    return (f(add_bias(X) @ w) > threshold).astype(int)


# Optional: real data.

def make_breast_cancer_split():
    """Return X_tr3, X_te3, t_tr3, t_te3 for the breast cancer problem."""
    X3, t3 = load_breast_cancer(return_X_y=True)
    return train_test_split(X3, t3, test_size=0.3, stratify=t3, random_state=0)


if __name__ == "__main__":
    X_tr, X_va, t_tr, t_va = make_gaussian_data()
    print(f"AND: X {X.shape}, targets {t}")
    print(f"Gaussian data: {len(X_tr)} train, {len(X_va)} validation")

    # Part 1: teaching a perceptron logical AND.
    print("\n=== Part 1: logical AND ===")
    w, history = perceptron_fit(X, t)
    print(f"Final weights (bias, w1, w2): {w}, epochs: {len(history)}")
    print(f"Mistakes per epoch: {history}")
    plot_boundary(lambda P: perceptron_predict(w, P), X, t, "Perceptron on AND",
                  save_as="part1_and_boundary.png")
    plot_history(history, "AND: mistakes per epoch", save_as="part1_and_history.png")

    # Epochs needed for each learning rate and starting weights (100 is the cap).
    etas = [0.01, 0.1, 1]
    starts = {"all zeros": None, "(1, 0, 0)": [1, 0, 0]}
    print("\nEpochs needed")
    print(f"{'Starting weights':18s}" + "".join(f"eta={eta:<8}" for eta in etas))
    for label, w_init in starts.items():
        epochs = [len(perceptron_fit(X, t, eta=eta, w_init=w_init)[1]) for eta in etas]
        print(f"{label:18s}" + "".join(f"{e:<12}" for e in epochs))

    # Part 2: a larger dataset with scikit-learn.
    print("\n=== Part 2: Gaussian data ===")
    clf = Perceptron(random_state=0)
    clf.fit(X_tr, t_tr)
    print(f"sklearn Perceptron: train accuracy {clf.score(X_tr, t_tr):.4f}, "
          f"validation accuracy {clf.score(X_va, t_va):.4f}")
    plot_boundary(clf.predict, X_tr, t_tr, "sklearn Perceptron", save_as="part2_sklearn_boundary.png")

    w2, history2 = perceptron_fit(X_tr, t_tr, max_epochs=1000)
    print(f"Own perceptron: {len(history2)} epochs, final-epoch mistakes {history2[-1]}, "
          f"minimum {min(history2)}, reached zero: {history2[-1] == 0}")
    print(f"Own perceptron: train accuracy {(perceptron_predict(w2, X_tr) == t_tr).mean():.4f}, "
          f"validation accuracy {(perceptron_predict(w2, X_va) == t_va).mean():.4f}")
    plot_history(history2, "Gaussian data: mistakes per epoch (own perceptron)",
                 save_as="part2_own_history.png")

    # Part 3: different activation functions.
    print("\n=== Part 3: activation functions ===")
    start = time.perf_counter()
    Perceptron(random_state=0).fit(X_tr, t_tr)
    rows = [("step (Part 2)", (time.perf_counter() - start) * 1000,
             clf.score(X_tr, t_tr), clf.score(X_va, t_va))]
    ws = {}
    for act in ACTIVATIONS:
        start = time.perf_counter()
        ws[act] = neuron_fit(X_tr, t_tr, act)
        elapsed = (time.perf_counter() - start) * 1000
        rows.append((act, elapsed,
                     (neuron_predict(ws[act], X_tr, act) == t_tr).mean(),
                     (neuron_predict(ws[act], X_va, act) == t_va).mean()))
    print(f"{'Activation':15s}{'Time (ms)':>10s}{'Train acc.':>12s}{'Val. acc.':>11s}")
    for name, ms, train_acc, val_acc in rows:
        print(f"{name:15s}{ms:10.1f}{train_acc:12.4f}{val_acc:11.4f}")
    for act in ACTIVATIONS:
        plot_boundary(lambda P, a=act: neuron_predict(ws[a], P, a), X_tr, t_tr,
                      f"{act} neuron", save_as=f"part3_{act}_boundary.png")

    # Optional: real data.
    print("\n=== Optional: breast cancer ===")
    X_tr3, X_te3, t_tr3, t_te3 = make_breast_cancer_split()
    raw = Perceptron(random_state=0).fit(X_tr3, t_tr3)
    scaled = make_pipeline(StandardScaler(), Perceptron(random_state=0)).fit(X_tr3, t_tr3)
    majority = np.bincount(t_tr3).argmax()
    print(f"Raw test accuracy:    {raw.score(X_te3, t_te3):.4f}")
    print(f"Scaled test accuracy: {scaled.score(X_te3, t_te3):.4f}")
    print(f"Always predicting the most common class ({majority}): "
          f"{(t_te3 == majority).mean():.4f}")
