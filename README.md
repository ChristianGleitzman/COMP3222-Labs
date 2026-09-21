# COMP3222-2026

Material for the 2026 labs for COMP3222 Machine Learning Technologies.

## Repository layout

| Directory | Contents |
| --- | --- |
| `sheets` | Lab sheet PDFs |
| `code/lab1` … `code/lab4` | Example/starter Python for each lab |
| `data/lab1` … `data/lab4` | Data files used by the labs |
| `notebooks` | Jupyter notebooks (`labW1_Intro_to_Jupyter.ipynb`, `lab1.ipynb`) |

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install numpy pandas scikit-learn numba matplotlib jupyter
```

Run the lab 1 examples with `python code/lab1/lab_1.py`, or work through
`notebooks/lab1.ipynb`, which demonstrates the same code cell by cell.

---

## Lab 1 (Week 2) — summary

Lab sheet: `sheets/Lab Sheet 1.pdf`. Based on the Week 1 lectures (L1.1, L1.2).
Example code: `code/lab1/lab_1.py`. Notebook walkthrough: `notebooks/lab1.ipynb`.

If you want NumPy practice first, work through `notebooks/labW1_Intro_to_Jupyter.ipynb`.

### Task 1 — Distance functions and Numba

Distance/similarity is fundamental to many ML algorithms. This task shows why
native Python loops are slow and how Numba fixes that. All functions take two
1D NumPy arrays of equal length and return a float; no input checks.

1. **Euclidean distance** (O(n)): a single loop summing squared differences,
   then a square root.

   $$d_e(a, b) = \sqrt{\sum_{i=1}^{m}(a_i - b_i)^2}$$

   Signature: `def euclidean_distance_python(x: np.ndarray, y: np.ndarray) -> float:`
2. **Numba version**: the same function under a different name, decorated with
   `@njit(cache=True)` (lecture 1.2, slide 40).
3. **Timing experiment**: time both over increasing `n` and plot time vs `n` on
   log–log axes with matplotlib. `n` has to be large — these functions are fast.
   The simplest timing pattern is `start = time.time()` … `time.time() - start`.

**Optional extension (do tasks 2 and 3 first)** — **Dynamic Time Warping**
(O(n²)), a time series distance that compensates for misalignment between two
series. Implement the standard algorithm: initialise an `(m+1) × (m+1)` cost
matrix `C` to infinity with `C[0, 0] = 0`, then for each `i, j` in `1…m`

```
C[i, j] = (x[i] - y[j])**2 + min(C[i-1, j-1], C[i-1, j], C[i, j-1])
```

returning `C[m, m]`. Re-run the timing experiment with and without Numba and
estimate the speed-up for both Euclidean and DTW.

### Task 2 — scikit-learn preprocessing and pipelines

The three preprocessing scenarios from Lecture 2, handled in scikit-learn.

- **A. Categorical variables** — `data/lab1/one_hot.csv` has three continuous
  variables (`x1, x2, x3`), one ordinal (`rank`), one nominal (`colour`) and the
  class label in the last column. Load with `pd.read_csv(..., header=None)`,
  build a numeric matrix by appending `rank` as a float column with
  `np.column_stack`, and one-hot encode `colour` with
  `sklearn.preprocessing.OneHotEncoder` (it expects 2D input, so reshape).
  Sanity check by fitting any classifier on the reformatted `X`.
  [Tutorial](https://inria.github.io/scikit-learn-mooc/python_scripts/03_categorical_pipeline.html)
- **B. Missing values** — `data/lab1/missing.csv` marks missing values with a
  blank cell or the text `NaN`. Load it and impute with something from
  `sklearn.impute`, e.g. `SimpleImputer`.
- **C. Scaling/standardisation** — standardise the imputed data with
  `StandardScaler` from `sklearn.preprocessing`.
- **D. Pipelines** — chain preprocessing and a model into a single estimator:

  ```python
  pipe = Pipeline([
      ("impute", SimpleImputer(strategy="median")),
      ("scale", StandardScaler()),
      ("clf", KNeighborsClassifier()),
  ])
  ```

  To go further, use `ColumnTransformer` to handle the mixed categorical and
  continuous `one_hot` data.
- **E. Train/test split** — always evaluate on held-out data with
  `train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)`, then
  `pipe.fit(X_train, y_train)` and `pipe.predict(X_test)`. More on evaluation in
  Week 5.

### Task 3 — Implement your own classifier

Write a scikit-learn compatible classifier that applies a simple decision rule on
a single feature. Binary classification only.

- **Constructor**: set the index of the feature to use (default 0).
- **`fit`**: find the mean of that feature for each class, and set the threshold
  to the median of all values of the feature. Whichever class has the lower mean
  becomes the "less than" prediction.
- **`predict`**: apply the rule — if the class 0 mean is the smaller one, predict
  class 0 when `value < threshold`, else class 1.

Following the scikit-learn conventions matters, so the classifier works with the
rest of the toolkit later in the module and in the coursework.
