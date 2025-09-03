import os
import pickle
import pandas as pd
from preprocess import preprocess_data
from train_validation_test import train_validation_test
from nn_from_scratch import NeuralNetwork   

# Paths
models_dir = os.path.join(os.getcwd(), "models")
reports_dir = os.path.join(os.getcwd(), "reports")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# 1. Load raw dataset
raw_file_path = os.path.join(os.getcwd(), "data", "dataset.csv")
df = pd.read_csv(raw_file_path)

# 2. Preprocess data
df_processed = preprocess_data(df)

# 3. Split features (X) and labels (y)
X = df_processed.drop(columns=["Churn"])  # features
y = df_processed["Churn"]                  # target

# 4. Split into train/test
X_train, X_test, y_train, y_test = train_validation_test(
    X, y, test_size=0.2, random_seed=42
)

# 5. Build neural network
nn = NeuralNetwork(
    layers_config = [
          {"units": X_train.shape[1], "activation": None},
          {"units": 64, "activation": "relu"},
          {"units": 32, "activation": "relu"},
          {"units": 1, "activation": "sigmoid" }
      ],
    loss="binary_crossentropy",
    lr=0.01,
    seed=42
)

# 6. Train neural network
nn.fit(X_train, y_train, epochs=100)

# 7. Evaluate on test set
y_pred = nn.predict(X_test)

# compute test loss using the same internal loss function
test_loss = nn.compute_loss(y_test, y_pred)

# compute accuracy
accuracy = (y_pred.round() == y_test).mean()

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")


with open(os.path.join(models_dir, "nn_scratch.pkl"), "wb") as f:
    pickle.dump(nn, f)

# Save results
results_path = os.path.join(reports_dir, "results.csv")
df_results = pd.DataFrame([{
    "Model": "NN (Scratch)",
    "Test Loss": test_loss,
    "Test Accuracy": accuracy
}])
df_results.to_csv(results_path, index=False, mode="w")


