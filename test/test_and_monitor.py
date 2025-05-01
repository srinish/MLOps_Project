import requests
import json
import pandas as pd
import whylogs as why
from whylogs.api.writer.whylabs import WhyLabsWriter

# Define the API endpoint
url = "http://127.0.0.1:5000/invocations"  # MLflow serving endpoint

# Test the model with new data
test_data = pd.DataFrame({
    "YearsExperience": [5, 10, 15]
    # Add other features if required to match the training data
})

# Convert the test data to JSON format
payload = {
    "inputs": test_data.to_dict(orient="records")
}

# Send the POST request
response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

# Check the response
if response.status_code == 200:
    predictions = response.json()
    print("Prediction:", predictions)

    # Log the input data and predictions using whylogs
    results = test_data.copy()
    results["Predictions"] = predictions  # Add predictions to the logged data
    profile = why.log(pandas=results)

    # Send the logs to WhyLabs
    writer = WhyLabsWriter(org_id="your_org_id", api_key="your_api_key")
    writer.write(profile.view(), dataset_id="your_dataset_id")
else:
    print(f"Error: {response.status_code}, {response.text}")