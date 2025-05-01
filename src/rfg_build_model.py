from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from src.model_utils import load_data, save_model, load_model, split_data
import yaml
import mlflow
import pandas as pd
from mlflow.sklearn import log_model
from mlflow.models import infer_signature
from mlflow import MlflowClient

def build_random_forest_model(X_train, y_train, params):
    """
    Build and train the Random Forest Regressor model.
    """
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_regressor_model(model, X_test, y_test):
    """
    Evaluate the regressor model's performance on the test set.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    signature = infer_signature(X_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    return mse, r2, signature

def assign_alias_to_stage(model_name, stage, alias):
    """
    Assign an alias to the latest version of a registered model within a specified stage.

    :param model_name: The name of the registered model.
    :param stage: The stage of the model version for which the alias is to be assigned. Can be
                "Production", "Staging", "Archived", or "None".
    :param alias: The alias to assign to the model version.
    :return: None
    """
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    print(mlflow.get_tracking_uri())
    # Initialize an MLflow Client
    client = MlflowClient()
    filter_string = f"name='{model_name}'"
    results = client.search_registered_models(filter_string=filter_string)
    print("-" * 80)
    for res in results:
        for mv in res.latest_versions:
            print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")
            client.set_registered_model_alias(model_name, alias, mv.version)
    print(client.get_registered_model(model_name))

# Update the main function
def main(): 
    # Define file paths
    data_file_path = '../data/dataset_v1.csv'
    model_file_path = '../random_forest_model.pkl'

    # Load parameters from YAML file
    with open("../config/model_params.yaml", "r") as file:
        params = yaml.safe_load(file)["random_forest"]

    # Load and preprocess data
    data = load_data(data_file_path)
    # data = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data, target_column='Salary')

    # Build and train the Random Forest model
    model = build_random_forest_model(X_train, y_train, params)

    # Evaluate the model
    mse, r2, signature  = evaluate_regressor_model(model, X_test, y_test)

    # Save the model
    save_model(model, model_file_path)
    # Log the experiment with MLflow
    # run_mlflow_experiment(model, X_train, y_train, X_test, y_test, params, model_file_path, data_file_path)
   
    # assign_alias_to_stage("sk-learn-random-forest-model", "Production", "champion")





if __name__ == "__main__":
    main()