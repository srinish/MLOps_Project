from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score,  mean_squared_error, r2_score
from model_utils import load_data, save_model, load_model, split_data
import yaml
import mlflow
import pandas as pd

def build_decision_tree_model(X_train, y_train, params):
    """
    Build and train the Decision Tree Regressor model.
    """
    model = DecisionTreeRegressor(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_regressor_model(model, X_test, y_test):
    """
    Evaluate the regressor model's performance on the test set.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    return mse, r2

# Update the main function
def main(): 
    # Define file paths
    data_file_path = './data/dataset_v1.csv'
    model_file_path = 'decision_tree_model.pkl'

    # Load parameters from YAML file
    with open("model_params.yaml", "r") as file:
        params = yaml.safe_load(file)["decision_tree"]

    # Load and preprocess data
    data = load_data(data_file_path)
    # data = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data, target_column='Salary')

    # Build and train the Decision Tree model
    model = build_decision_tree_model(X_train, y_train, params)

    # Evaluate the model
    mse, r2 = evaluate_regressor_model(model, X_test, y_test)

    # Save the model
    save_model(model, model_file_path)


    # Log the experiment with MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    print(mlflow.get_tracking_uri())
    mlflow.set_experiment("Decision_Tree_Experiment-V1")

    with mlflow.start_run():
        mlflow.log_param("model_type", "Decision Tree Regressor")
        mlflow.log_param("max_depth", params.get("max_depth", None))
        mlflow.log_param("min_samples_split", params.get("min_samples_split", 2))
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_artifact(model_file_path, artifact_path="models")
        mlflow.log_artifact(data_file_path, artifact_path="data")
        mlflow.log_artifact("model_params.yaml", artifact_path="params")
        mlflow.log_artifact("dcm_build_model.py", artifact_path="scripts")

        # Log the model
        mlflow.sklearn.log_model(model, "decision_tree_model")
        mlflow.end_run()

if __name__ == "__main__":
    main()