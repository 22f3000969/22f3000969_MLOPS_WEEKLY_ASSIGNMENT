from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os
import json

# Load training data
df = pd.read_csv("data/iris_train.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Evaluate on training data (quick sanity metric for this assignment)
train_preds = model.predict(X)
train_acc = accuracy_score(y, train_preds)

# Evaluate on test data if available
test_acc = None
if os.path.exists("data/iris_test.csv"):
    test_df = pd.read_csv("data/iris_test.csv")
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]
    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)

# Save model to a STABLE path (DVC + Git tags handle versioning, not folder timestamps)
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

# Save metrics for reference (also useful for the report/screencast)
metrics = {
    "train_rows": len(df),
    "train_accuracy": round(train_acc, 4),
    "test_accuracy": round(test_acc, 4) if test_acc is not None else None
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Training completed")
print("Model saved to: model/model.pkl")
print("Metrics:", metrics)
