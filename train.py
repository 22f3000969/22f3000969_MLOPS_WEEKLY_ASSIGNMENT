
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
from datetime import datetime
import os

# Load training data
df = pd.read_csv("iris_train.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Create timestamp folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

artifact_dir = f"artifacts/{timestamp}"
os.makedirs(artifact_dir, exist_ok=True)

# Save model
joblib.dump(model, f"{artifact_dir}/model.pkl")

print("Training completed")
print("Artifacts stored in:", artifact_dir)
