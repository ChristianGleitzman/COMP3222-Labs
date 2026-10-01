import pandas as pd
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "lab2" / "playgolf.csv"

if __name__ == "__main__":    # Example usage
    data = pd.read_csv(DATA_PATH)
    print(data)
    y = data.iloc[:, 4].to_numpy()
    print(y)
    X = data.iloc[:, 0:4].to_numpy()
    print(X.shape)
    outlook = X[:, 0]
    print(outlook)
    print(outlook.shape)
