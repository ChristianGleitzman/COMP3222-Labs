# Lab 7: neural networks for classification and regression

The exercises are in `Lab7.pdf` in the repository root. Run this from the repository root:

```bash
python code/lab7/lab7.py
```

This is a **student starter**. The data loading and 80/20 splits, the two parameter grids, `show` and the imports are provided. The functions marked `NotImplementedError` and the steps listed as `# TODO` in the `__main__` block are yours. Figures saved with `save_as` go to `code/lab7/results`, which `.gitignore` excludes.

| Part | Steps |
|---|---|
| A: breast cancer | B1 logistic regression baseline, B2 `MLPClassifier`, B3 grid search, B4 missed tumours and the decision threshold |
| B: California housing | C1 linear regression baseline, C2 `MLPRegressor` and training curves, C3 grid search, C4 predicted against actual |

Two things differ from Lab 5. The breast cancer labels are **not** recoded: 0 is malignant and 1 is benign. The California housing data is downloaded on the first run, so it needs an internet connection once.
