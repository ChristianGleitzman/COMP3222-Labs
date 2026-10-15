"""Lab 3, Task 2 solution: compare regression models."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from lab3_task2 import load_wine_quality_data


def report(name, model, X_train, X_test, y_train, y_test):
    """Fit the model and print its training and test MSE."""
    model.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"{name:22s} train MSE {train_mse:.3f}  test MSE {test_mse:.3f}")


if __name__ == "__main__":
    X, y = load_wine_quality_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42
    )
    data = (X_train, X_test, y_train, y_test)

    report("LinearRegression", LinearRegression(), *data)
    for depth in [1, 3, 5, 8, 12, None]:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
        report(f"Tree max_depth={depth}", tree, *data)
    report("RandomForestRegressor", RandomForestRegressor(random_state=42), *data)
