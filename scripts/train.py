import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os
import mlflow
from mlflow import sklearn
import subprocess
import time

# Load the dataset directly from the file path (data.csv is now committed to Git)
data_path = 'data/data.csv'
data = pd.read_csv(data_path)

# Assuming the data has columns "features" and "target"
X = data.drop(columns='target')  # features
y = data['target']  # target variable

# Split the data into training and testing sets (adjust as necessary)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Set the experiment name
mlflow.set_experiment("exp_name2")

# Enable MLflow autologging
mlflow.sklearn.autolog()

# Model training and logging
with mlflow.start_run() as run:
    # Initialize the model (Logistic Regression in this case)
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate model accuracy on the test set
    accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")

    # Log custom metrics
    mlflow.log_metric("test_accuracy", accuracy)

    # Save the trained model to a file (local backup)
    try:
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/trained_model.pkl')
        print("Model training complete and saved.")
    except Exception as e:
        print(f"Error saving model: {e}")

# Stop the MLflow server (if started programmatically)
mlflow_server.terminate()
mlflow_server.wait()
print("MLflow server stopped.")
