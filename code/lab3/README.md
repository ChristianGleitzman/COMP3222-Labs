# Lab 3 (Week 4): encoding, regression and ensembles

Read [Lab Sheet 3](../../Lab%20Sheet%203.pdf) for the exercises. Run these commands from the repository root after installing the packages in the [main README](../../README.md):

```bash
python code/lab3/lab3_task1.py
python code/lab3/lab3_task2.py
python code/lab3/lab3_task3.py
```

These are **student starters**. The data loading and estimator constructors run, but the functions marked `NotImplementedError` are yours to complete.

## Task 1: encode categorical features

Implement `frequency_encode` and `impact_encode` in `lab3_task1.py`. The former uses the proportion of rows in each category. For binary labels 0 and 1, the latter uses the mean label within each category. After implementing them, call `check_functions()` to compare with the Week 3 lecture example.

Then implement `FrequencyEncoderTransformer` and `ImpactEncoderTransformer`. In `fit`, learn category mappings **from the training rows only**. In `transform`, apply those mappings to both training and new rows, leave numeric columns numeric, and return a float array. The sheet specifies simple values for categories unseen during fitting.

`load_credit_approval_data()` loads `data/lab3/credit_approval.csv` and drops incomplete rows. This keeps the exercise focused on encoding; Lab 1 covered imputation. Split the mixed feature array before fitting a transformer if you evaluate on unseen cases. For impact encoding, use the labels only in `fit` and avoid learning mappings from the test set.

## Task 2: regression

`lab3_task2.py` loads `data/lab3/winequality-red.csv` and makes a reproducible 70/30 train/test split. The `quality` column is the target. Complete the marked comparisons using `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor` and `mean_squared_error`. Print both training and test MSE for every model. Vary tree depth to see how overfitting changes the two errors.

The red-wine data is from the [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality), by Cortez and colleagues (2009), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The repository keeps the original semicolon-delimited CSV and `quality` scores.

## Task 3: majority vote

Complete `BasicEnsembleClassifier.fit` and `.predict` in `lab3_task3.py`. Its constructor and scikit-learn parameters are ready to use. Fit each cloned member on a random subset, then take a majority vote for each new row. Put the Task 1 transformer and your classifier in a `Pipeline` so fitting the pipeline learns its encoder from the training rows only.

The optional comparison uses `RandomForestClassifier` from scikit-learn and `RotationForestClassifier` from `aeon.classification.sklearn`. The [main README](../../README.md) includes `aeon` in the install command.
