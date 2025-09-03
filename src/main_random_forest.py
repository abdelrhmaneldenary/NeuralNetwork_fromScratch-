# main.random_forest.py

import os
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss



# Paths
models_dir = os.path.join(os.getcwd(), "models")
reports_dir = os.path.join(os.getcwd(), "reports")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# 1. Load processed dataset (same as Keras file for fairness)
processed_file_path = os.path.join(os.getcwd(), "data", "processed_dataset.csv")
df = pd.read_csv(processed_file_path)

# 2. Features & target
X = df.drop(columns=["Churn"]).values
y = df["Churn"].values

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# 4. Build Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,        # number of trees
    max_depth=None,          # let trees grow fully
    random_state=42,         # reproducibility
    n_jobs=-1                # use all CPU cores
)

# 5. Train model
rf_model.fit(X_train, y_train)

# 6. Predict
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]  # needed for log-loss

# 7. Evaluate
acc = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_proba)

print(f"Test Accuracy: {acc:.4f}")
print(f"Test Log Loss: {loss:.4f}")


# Save model
with open(os.path.join(models_dir, "rf_model.pkl"), "wb") as f:
    pickle.dump(rf_model, f)

# Save results (append)
results_path = os.path.join(reports_dir, "results.csv")
df_results = pd.DataFrame([{
    "Model": "Random Forest",
    "Test Loss": loss,
    "Test Accuracy": acc
}])
df_results.to_csv(results_path, index=False, mode="a", header=False)