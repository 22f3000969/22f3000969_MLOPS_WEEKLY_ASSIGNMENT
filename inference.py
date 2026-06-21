
import pandas as pd
import joblib
import glob

# Find latest model
models = sorted(glob.glob("artifacts/*/model.pkl"))
latest_model = models[-1]

print("Using model:", latest_model)

model = joblib.load(latest_model)

# Load test data
df = pd.read_csv("iris_test.csv")

X = df.drop("target", axis=1)

predictions = model.predict(X)

output = pd.DataFrame({
    "prediction": predictions
})

output.to_csv("predictions.csv", index=False)

print("Inference completed")
print("Predictions saved to predictions.csv")
