from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import pickle
import os
import mlflow 
import yaml
from src.model_utils import load_data, save_model, load_model, split_data
from mlflow.sklearn import log_model
from mlflow.models import infer_signature

def build_model(X_train, y_train, params):
    """
    Build and train the logistic regression model.
    """
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance on the test set.
    """ 
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    signature = infer_signature(X_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    return accuracy, signature


# Define the main function
def main(): 
    # Define file paths
    data_file_path = '../data/dataset_v1.csv'
    model_file_path = '../models/logistic_regression_model.pkl'

    # Load parameters from YAML file
    with open("../config/model_params.yaml", "r") as file:
        params = yaml.safe_load(file)["logistic_regression"]

    # Load and preprocess data
    data = load_data(data_file_path)
    #data = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data, target_column='Salary')

    # Build and train the model
    model = build_model(X_train, y_train, params)

    # Evaluate the loaded model
    accuracy, signature = evaluate_model(model, X_test, y_test)

    # Save the model
    save_model(model, model_file_path)
 
   

    """
    Run an MLflow experiment to log the model and its parameters.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    print(mlflow.get_tracking_uri())
    # setup mlflow experiment
    mlflow.set_experiment("Logistic_Regression_Experiment-V1")
   
   
    # log parameters and metrics
    with mlflow.start_run():
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", accuracy)  # Example metric, replace with actual value
        mlflow.log_artifact(model_file_path, artifact_path="models")
        mlflow.log_artifact(data_file_path, artifact_path="data")
        mlflow.log_artifact("model_params.yaml", artifact_path="params")
        mlflow.log_artifact("build_model.py", artifact_path="scripts")

        mlflow.sklearn.log_model(model, "Logistic_Regression_Model", signature=signature)

    
  
if __name__ == "__main__":
    main()
# This script builds a logistic regression model, evaluates its performance, and saves it to a file.