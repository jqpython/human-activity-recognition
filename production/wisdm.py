import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--trainingdata", type=str, required=True, help="Path to the WISDM dataset CSV file"
)
args = parser.parse_args()

# Load and preprocess dataset
try:
    # Read the WISDM dataset
    data = pd.read_csv(args.trainingdata)

    # Verify the expected columns are present
    expected_columns = [
        "user_id",
        "activity",
        "timestamp",
        "x_accel",
        "y_accel",
        "z_accel",
    ]
    if not all(col in data.columns for col in expected_columns):
        raise ValueError(f"Dataset must contain these columns: {expected_columns}")

except Exception as e:
    print(f"Error processing the dataset: {e}")
    exit(1)

# Data preprocessing
print("Preprocessing data...")

# Create feature matrix X and target vector y
X = data[
    ["x_accel", "y_accel", "z_accel"]
]  # Using only accelerometer readings as features
y = data["activity"]

# Check for and handle missing values
if X.isnull().any().any():
    print("Warning: Dataset contains missing values. Dropping missing rows.")
    # Get indices where any of X columns are null
    null_indices = X.isnull().any(axis=1)
    X = X[~null_indices]
    y = y[~null_indices]

# Split the dataset
print("Splitting dataset into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # Ensure balanced split across activities
)

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1  # Use all available cores
)
model.fit(X_train_scaled, y_train)

# Evaluate model
print("Evaluating model...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
class_report = classification_report(y_test, y_pred)
print(class_report)

# Create and save confusion matrix plot
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
activities = sorted(y.unique())
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

# Add activity labels to the plot
tick_marks = np.arange(len(activities))
plt.xticks(tick_marks, activities, rotation=45)
plt.yticks(tick_marks, activities)

# Add number labels to the cells
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# Log with MLflow
print("\nLogging results with MLflow...")
mlflow.log_metric("accuracy", accuracy)
mlflow.log_artifact("confusion_matrix.png")
mlflow.sklearn.log_model(model, "model")

# Save the scaler for future use
print("Saving scaler...")
mlflow.sklearn.log_model(scaler, "scaler")

print("\nTraining and evaluation complete!")
