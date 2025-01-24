import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import mlflow
import subprocess
import os
import time

# Start the MLflow server programmatically
mlflow_server_command = [
    "mlflow", "server",
    "--backend-store-uri", "file:///D:/Automation_pipeline/mlruns2",
    "--default-artifact-root", "file:///D:/Automation_pipeline/mlarti",
    "--host", "0.0.0.0",
    "--port", "5000"
]

new_artifact_root = "file:///D:/Automation_pipeline/mlarti"

print("Starting the MLflow server...")
mlflow_server = subprocess.Popen(mlflow_server_command)

time.sleep(5)  # Wait for the server to initialize

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")  # Replace with your MLflow server URI if needed

# Override the artifact root for the new run
os.environ["MLFLOW_ARTIFACT_URI"] = new_artifact_root

mlflow.set_experiment("exp_name2")  # Ensure this matches your training experiment

# Start a new MLflow run
with mlflow.start_run() as run:
    # Load the dataset
    data_path = 'data/data.csv'
    data = pd.read_csv(data_path)

    # Assuming the data has columns "features" and "target"
    X = data.drop(columns='target')  # features
    y = data['target']  # target variable

    # Load the saved model
    model = joblib.load('models/trained_model.pkl')

    # Predict and evaluate on the test set
    predictions = model.predict(X)

    # Calculate metrics
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, average='binary')  # Adjust 'binary' to 'micro', 'macro', or 'weighted' as needed
    recall = recall_score(y, predictions, average='binary')
    f1 = f1_score(y, predictions, average='binary')
    conf_matrix = confusion_matrix(y, predictions)

    # Print metrics
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    # Optionally, save the results
    with open('models/accuracy.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")

    # Log metrics to MLflow
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Create a confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save the figure locally and log it to MLflow
    conf_matrix_path = "models/confusion_matrix.png"
    plt.savefig(conf_matrix_path)
    plt.close()

    mlflow.log_artifact(conf_matrix_path)

    # Save the results locally
    with open('models/test_metrics.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Precision: {precision * 100:.2f}%\n")
        f.write(f"Recall: {recall * 100:.2f}%\n")
        f.write(f"F1 Score: {f1 * 100:.2f}%\n")
    mlflow.log_artifact("models/test_metrics.txt")

    # Exit with status based on the test result (to gate model registration)
    if accuracy < 0.8:  # Example threshold
        raise ValueError("Model accuracy is below acceptable threshold.")

# Stop the MLflow server (if started programmatically)
mlflow_server.terminate()
mlflow_server.wait()
print("MLflow server stopped.")
