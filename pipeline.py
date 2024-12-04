# pipeline.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Step 1: Load and Prepare Data
try:
    print("Loading dataset...")
    data = pd.read_csv('london_weather.csv')
    if data.isnull().sum().any():
        print("Handling missing values...")
        data.fillna(method='ffill', inplace=True)
    X = data.drop(columns=['mean_temp'])
    y = data['mean_temp']
except FileNotFoundError:
    print("Error: 'london_weather.csv' not found. Ensure the dataset is in the working directory.")
    exit()

# Step 2: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data successfully split into training and testing sets.")

# Step 3: Initialize MLflow
mlflow.set_experiment("Temperature Prediction")
experiment_results = []

# Step 4: Define Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Step 5: Train and Log Models
print("Training and logging models...")
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train Model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Evaluate Model
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        # Log Results with MLflow
        mlflow.log_param("model", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        # Save Results
        experiment_results.append({"model": model_name, "rmse": rmse})
        print(f"Model: {model_name}, RMSE: {rmse:.2f}")

# Step 6: Display Results
print("\nExperiment Results:")
for result in experiment_results:
    print(f"Model: {result['model']}, RMSE: {result['rmse']:.2f}")

print("Pipeline execution complete.")
