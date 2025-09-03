import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score
import os



models_dir = os.path.join(os.getcwd(), "models")
reports_dir = os.path.join(os.getcwd(), "reports")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

processed_file_path = os.path.join(os.getcwd(), "data", "processed_dataset.csv")
df = pd.read_csv(processed_file_path)

X = df.drop(columns=["Churn"]).values
y = df["Churn"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # input layer
    layers.Dense(64, activation='relu'),      # hidden layer
    layers.Dense(32, activation='relu'),      # hidden layer
    layers.Dense(1, activation='sigmoid')     # output layer (binary classification)
])


model.compile(
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    verbose=1
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# Predictions
y_pred = (model.predict(X_test) >= 0.5).astype(int)
print("Accuracy (sklearn):", accuracy_score(y_test, y_pred))


model.save(os.path.join(models_dir, "nn_keras.h5"))

# Save results (append to existing)
results_path = os.path.join(reports_dir, "results.csv")
df_results = pd.DataFrame([{
    "Model": "NN (Keras)",
    "Test Loss": loss,
    "Test Accuracy": acc
}])
df_results.to_csv(results_path, index=False, mode="a", header=False)