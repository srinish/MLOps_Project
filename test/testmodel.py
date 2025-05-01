import requests
import json
import pandas as pd

# Define the API endpoint
url = "http://127.0.0.1:5000/invocations"  # Correct MLflow serving endpoint

# Test the model with new data
test_data = pd.DataFrame({
    "YearsExperience": [5, 10, 15]
    # Add other features if required to match the training data
})

# Convert the test data to JSON format (MLflow expects "inputs" field in the new protocol)
payload = {
    "inputs": test_data.to_dict(orient="records")
}

# Send the POST request
response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

# Print the response
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print(f"Error: {response.status_code}, {response.text}")