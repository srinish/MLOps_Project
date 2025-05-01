import mlflow
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import mlflow

def save_model(model, model_path):
    """
    Save the trained model to a file.
    """
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
def load_model(model_path):
    """
    Load the trained model from a file.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model
def split_data(data, target_column):
    """
    Split the data into training and testing sets.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    return data

def run_mlflow_experiment(model, X_train, y_train, X_test, y_test, params, model_file_path, data_file_path):
    """
    Run an MLflow experiment to log the model and its parameters.
    """
 # Log the experiment with MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    print(mlflow.get_tracking_uri())
    mlflow.set_experiment("Random_Forest_Experiment-V1")

    with mlflow.start_run():
        mlflow.log_param("model_type", "Random Forest Regressor")
        mlflow.log_param("n_estimators", params.get("n_estimators", 100))
        mlflow.log_param("max_depth", params.get("max_depth", None))
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_artifact(model_file_path, artifact_path="models")
        mlflow.log_artifact(data_file_path, artifact_path="data")
        mlflow.log_artifact("../config/model_params.yaml", artifact_path="params")
        mlflow.log_artifact("rfg_build_model.py", artifact_path="scripts")

        # Log the model
        mlflow.sklearn.log_model(sk_model=model,
            artifact_path="random_forest_model",
            signature=signature,
            registered_model_name="sk-learn-random-forest-model")

        mlflow.end_run()