"""
Simulates new IRIS data arriving for a later pipeline iteration.
Generates additional synthetic-but-realistic IRIS rows by sampling
from the sklearn IRIS dataset's untouched portion, and appends them
to data/iris_train.csv — mimicking a real-world "more labeled data
became available" scenario between training iterations.
"""
from sklearn.datasets import load_iris
import pandas as pd
import sys

N_NEW_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 30

iris = load_iris(as_frame=True)
full_df = iris.frame.rename(columns={"target": "target"})

existing = pd.read_csv("data/iris_train.csv")
existing_test = pd.read_csv("data/iris_test.csv") if __import__("os").path.exists("data/iris_test.csv") else pd.DataFrame()

# Find rows from the full IRIS set not already used in train or test
combined_existing = pd.concat([existing, existing_test])
merge_cols = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]

unused = full_df.merge(combined_existing[merge_cols], on=merge_cols, how="left", indicator=True)
unused = unused[unused["_merge"] == "left_only"].drop(columns=["_merge"])

if len(unused) == 0:
    print("No unused IRIS rows left to add — all 150 rows already in train/test.")
    sys.exit(0)

n_to_add = min(N_NEW_ROWS, len(unused))
new_rows = unused.sample(n=n_to_add, random_state=None)

updated = pd.concat([existing, new_rows], ignore_index=True)
updated.to_csv("data/iris_train.csv", index=False)

print(f"Added {n_to_add} new rows to data/iris_train.csv")
print(f"New training set size: {len(updated)} rows (was {len(existing)})")
