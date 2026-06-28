"""
Simulates new labeled IRIS data arriving for a later pipeline iteration.
Moves a sample of rows from data/iris_test.csv into data/iris_train.csv,
mimicking a real-world scenario where previously held-out/unlabeled
samples get confirmed and folded into the training set.
"""
import pandas as pd
import sys
import os

N_NEW_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 15

train_path = "data/iris_train.csv"
test_path = "data/iris_test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

if len(test_df) <= N_NEW_ROWS:
    print(f"Not enough test rows left to move ({len(test_df)} remaining). Reduce N_NEW_ROWS.")
    sys.exit(1)

# Randomly select rows from test to "promote" into train
moved_rows = test_df.sample(n=N_NEW_ROWS, random_state=None)
remaining_test = test_df.drop(moved_rows.index)

updated_train = pd.concat([train_df, moved_rows], ignore_index=True)

updated_train.to_csv(train_path, index=False)
remaining_test.to_csv(test_path, index=False)

print(f"Moved {N_NEW_ROWS} rows from test set into training set")
print(f"New training set size: {len(updated_train)} rows (was {len(train_df)})")
print(f"New test set size: {len(remaining_test)} rows (was {len(test_df)})")
