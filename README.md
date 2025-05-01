# MLOps Project: Model Deployment and Monitoring

This project demonstrates the end-to-end process of building, deploying, and monitoring machine learning models using **MLflow** and **WhyLabs**. The project includes training a model, deploying it in a production-like environment, and monitoring its performance for operational drifts and anomalies.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Monitoring with WhyLabs](#model-monitoring-with-whylabs)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Project Overview
This project focuses on:
1. Training machine learning models (e.g., Logistic Regression, Random Forest, Decision Tree).
2. Deploying the best-performing model using **MLflow**.
3. Monitoring the deployed model in production using **WhyLabs** to detect drifts and anomalies.

---

## Features
- **Model Training**: Train multiple models and evaluate their performance.
- **Model Deployment**: Deploy the best-performing model using MLflow's model serving capabilities.
- **Model Monitoring**: Monitor the deployed model's performance using WhyLabs.
- **Drift Detection**: Detect data and concept drifts in production.
- **Anomaly Detection**: Identify anomalies in input data and predictions.

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- MLflow installed (`pip install mlflow`)
- WhyLabs SDK installed (`pip install whylabs-client whylogs`)
- Docker (optional, for containerized deployment)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/MLOps_Project.git
   cd MLOps_Project

2.Install dependencies:

pip install -r requirements.txt
3. Set up MLflow:

    Start the MLflow tracking server:
    Open the MLflow UI at http://127.0.0.1:5000.
4. Set up WhyLabs:

    Create an account at WhyLabs.
    Note your Organization ID, Dataset ID, and API Key.

Usage
1. Train and Evaluate Models
    Run the build_model.py script to train and evaluate models:

2. Deploy the Best Model
    Deploy the best-performing model using MLflow:

3. Test the Deployed Model
    Use the testmodel.py script to send test data to the deployed model:

    python testmodel.py
