# Lab 1 (Week 2): distance functions and preprocessing

This lab follows the Week 1 lectures. Read [Lab Sheet 1](../../Lab%20Sheet%201.pdf) for the full instructions. The [Lab 1 notebook](../../notebooks/lab1.ipynb) walks through the starter examples and leaves the exercises for you to complete. For earlier NumPy practice, use the [Jupyter introduction](../../notebooks/labW1_Intro_to_Jupyter.ipynb).

From the repository root, run the starter example with:

```bash
python code/lab1/lab_1.py
```

The three worked solutions are `lab1_task1_solution.py`, `lab1_task2_solution.py`, and `lab1_task3_solution.py` in this directory. They show one way to complete the tasks after you have attempted them.

## Task 1: Euclidean distance and Numba

Write a function that takes two equally long 1D NumPy arrays. In one loop, add the squared difference at each position, then return the square root of the sum:

$$d(x,y)=\sqrt{\sum_{i=1}^{m}(x_i-y_i)^2}.$$

Copy the function under a second name and add `@njit(cache=True)`. Check that both functions give the same answer. Time each over increasing input lengths, **after a warm-up call to compile the Numba function**, and plot time against length on log–log axes. `lab_1.py` demonstrates the timing and plotting pattern.

The optional Dynamic Time Warping (DTW) extension uses a cost matrix `C` of shape `(m + 1, m + 1)`, filled with infinity except `C[0, 0] = 0`. For `i, j` from 1 to `m`, compute:

```python
C[i, j] = (x[i - 1] - y[j - 1])**2 + min(
    C[i - 1, j - 1], C[i - 1, j], C[i, j - 1]
)
```

Return `C[m, m]`. DTW takes quadratic time and memory, so benchmark it at much smaller lengths than Euclidean distance.

## Task 2: preprocessing and pipelines

- **Categorical data:** `data/lab1/one_hot.csv` has three continuous features, an ordinal `rank`, a nominal `colour`, and a final class label. Read it with `header=None`. Add `rank` as a numeric column with `np.column_stack`, then one-hot encode `colour` (reshape it to 2D first). Check that a classifier can fit the resulting numeric matrix.
- **Missing values:** `data/lab1/missing.csv` uses blank cells and `NaN` for missing values. Load it with pandas and fill feature values with `SimpleImputer`.
- **Scaling:** Use `StandardScaler` on the imputed numeric features.
- **Pipeline:** Chain imputation, scaling, and a classifier. A `KNeighborsClassifier` makes scaling relevant because it uses distances.
- **Evaluation:** Split the original feature matrix into training and test sets first. Fit the entire pipeline on the training set, then predict on the test set. This keeps the imputer and scaler from learning from test data.

For mixed categorical and numeric data, `ColumnTransformer` is an optional extension. [This scikit-learn tutorial](https://inria.github.io/scikit-learn-mooc/python_scripts/03_categorical_pipeline.html) gives another example.

## Task 3: a single-feature classifier

Build a binary classifier following scikit-learn's estimator pattern. The constructor stores the selected feature index (default `0`). In `fit`, find that feature's mean for each class and its median over all training rows. The class with the lower mean is predicted for values **below** the median; the other class is predicted at or above it. In `predict`, apply this learned rule to every row.

Keep values learned during `fit` in attributes ending in `_`, such as `threshold_` and `classes_`. See `lab1_task3_solution.py` after trying the exercise.
