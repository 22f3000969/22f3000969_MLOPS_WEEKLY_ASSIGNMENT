# 22F3000969 — MLOps Weekly Assignment (Week 2): DVC Integration

This branch (`week_2`) extends the Week 1 IRIS classification pipeline with
**Data Version Control (DVC)**, backed by **Google Cloud Storage (GCS)** as
the remote, to make data and model artifacts fully versioned and reproducible.

## What's in this repo

| File | Purpose |
|---|---|
| `train.py` | Trains a RandomForest classifier on `data/iris_train.csv`, evaluates on `data/iris_test.csv`, saves model to `model/model.pkl` and metrics to `metrics.json` |
| `inference.py` | Loads the latest trained model and runs predictions |
| `augment_data.py` | Simulates new labeled data arriving by moving rows from the test set into the training set, for iteration 2 |
| `data/iris_train.csv.dvc`, `data/iris_test.csv.dvc` | DVC pointer files for the versioned datasets |
| `model/model.pkl.dvc` | DVC pointer file for the versioned trained model |
| `.dvc/config` | DVC remote configuration (GCS bucket) |
| `requirements.txt` | Python dependencies, including `dvc` and `dvc-gs` |

## Setup (for a reviewer / fresh clone)

```bash
git clone <repo-url>
cd 22f3000969_MLOPS_WEEKLY_ASSIGNMENT
git checkout week_2

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pull versioned data/model from GCS remote
dvc pull
```

## DVC Remote

Configured as a default remote pointing to a GCS bucket:

```
gs://22f3000969-dvc-iris-remote
```

## Versioning Workflow Used

### Iteration 1 — Baseline (tag `v1.0`)
- 120 training rows, 30 test rows
- `dvc add data/iris_train.csv data/iris_test.csv`
- `python train.py` → train accuracy 1.0, test accuracy 0.9
- `dvc add model/model.pkl`
- Committed and tagged `v1.0`

### Iteration 2 — Augmented (tag `v2.0`)
- Simulated new labeled data by moving 15 rows from test → train via `augment_data.py`
- 135 training rows, 15 test rows
- `dvc add data/iris_train.csv data/iris_test.csv` (re-tracked, new hashes)
- `python train.py` → train accuracy 1.0, test accuracy 1.0
- `dvc add model/model.pkl` (re-tracked, new hash)
- Committed and tagged `v2.0`

## Switching Between Versions (Task 4 demo)

```bash
# Go back to the baseline version
git checkout v1.0
dvc checkout
# data/iris_train.csv reverts to 120 rows, model reverts to v1.0 weights

# Move forward to the augmented version
git checkout v2.0
dvc checkout
# data/iris_train.csv becomes 135 rows, model updates accordingly

# Return to latest
git checkout week_2
dvc checkout
```

Each `dvc checkout` swaps the actual file content (data + model) to match
the `.dvc` pointers at that Git revision — verified by checking row counts
in the CSVs and the accuracy values in `metrics.json` after each switch.

## Key Commands Reference

| Command | Purpose |
|---|---|
| `dvc init` | Initialize DVC in the Git repo |
| `dvc remote add -d gcs_remote gs://<bucket>` | Configure default GCS remote |
| `dvc add <file>` | Start tracking a file with DVC, creates `.dvc` pointer |
| `dvc push` | Upload tracked data/model to the GCS remote |
| `dvc pull` | Download tracked data/model from the GCS remote |
| `dvc checkout` | Sync workspace files to match `.dvc` pointers at current Git commit |
| `git tag -a vX.0 -m "..."` | Tag a commit as a named version milestone |
