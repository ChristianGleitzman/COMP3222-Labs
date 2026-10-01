# Lab 2 (Week 3): decision trees

This lab follows the Week 2 lectures. Read [Lab Sheet 2](../../Lab%20Sheet%202.pdf) for the full exercises. It moves from calculating split quality by hand to fitting and interpreting scikit-learn decision trees.

Run commands from the repository root. Install the core packages in the [root setup guide](../../README.md). The ItalyPowerDemand loader in Task 3 also needs:

```bash
python -m pip install aeon
```

## Task 1: Gini impurity and gain

Start with `data/lab2/playgolf.csv`. The starter file `lab2_task1.py` shows how to load its 14 rows, four attributes, and class labels:

```bash
python code/lab2/lab2_task1.py
```

Implement `impurity(counts)` for one node:

$$I(Z)=1-\sum_i p_i^2,$$

where `p_i` is the fraction of cases in class `i`. Then implement `gini_gain(attr, y)`: subtract the size-weighted impurity of each attribute group from the root impurity. `np.unique(..., return_counts=True)` helps with class counts. `Outlook` contains strings; encode its categories as integers if your gain function requires integer input.

For a check, the supplied data has root impurity about **0.4592**. Its Gini gains are about **0.1163** for Outlook, **0.0163** for Temp, **0.0274** for Humidity, and **0.0306** for Windy. Outlook gives the largest gain.

## Task 2: fit and view a tree

Use `lab2_task2.py` as the starter. It loads PlayGolf and marks the steps to complete:

```bash
python code/lab2/lab2_task2.py
```

Try `DecisionTreeClassifier` on the original features and inspect the error: scikit-learn cannot use the string values in `Outlook` directly. One-hot encode that column, leaving the other three numeric columns as they are. The resulting feature matrix has shape `(14, 6)` and the labels have shape `(14,)`. `ColumnTransformer` and `Pipeline` can keep the encoding with the classifier.

Fit a tree and use `sklearn.tree.plot_tree` with meaningful feature and class names. Compare its shape with the lecture's categorical tree. Try `ExtraTreeClassifier` as well. A tree's training accuracy alone does not tell you how well it will perform on new data.

The sheet also uses the helper `plot_tree_2d` from `lab2_task3.py` to display decision regions for two Iris features. That plot is separate from the PlayGolf tree and does not need `aeon`. In a notebook started at the repository root, import it with:

```python
import sys
sys.path.insert(0, "code/lab2")
from lab2_task3 import plot_tree_2d
```

## Task 3: control tree complexity

Use the ItalyPowerDemand time-series data to inspect `DecisionTreeClassifier` parameters. Run `python code/lab2/lab2_task3.py` to load the data and fit a first tree. The sheet shows how to load it with `aeon.datasets.load_italy_power_demand`; a copy of the dataset is also in `data/lab2/ItalyPowerDemand.ts`. Explore `criterion`, `splitter`, and then regularisation settings such as `max_depth`, `min_samples_leaf`, `max_leaf_nodes`, and `ccp_alpha`. Plot the trees or compare their depth and leaf counts to see how each setting changes the model. Set `random_state` when you want reproducible comparisons.

The sheet has two loader typos: use `return_type="numpy2d"` (lowercase `d`) for the built-in loader, and use `"ItalyPowerDemand"` (one final `d`) with the alternative `load_classification` function.
