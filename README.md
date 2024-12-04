# Temperature Prediction ML-Ops Pipeline For SlashUp
## Overview

This project aims to build a Machine Learning (ML) pipeline to predict the mean temperature in London using regression models. The objective is to ensure that the model achieves a Root Mean Square Error (RMSE) of less than or equal to 3.

The pipeline leverages MLflow for experiment tracking and model logging, making it easier to track the performance of multiple models and their corresponding hyperparameters. The project is implemented in Python and uses Scikit-learn for model training.
### Objective

    Build and train multiple regression models for predicting London's mean temperature.
    Use the london_weather.csv dataset to train and test the models.
    Track the models and their performance using MLflow.
    Ensure that the final model achieves an RMSE of ≤ 3.

### Requirements

    Software:
        Python 3.9+
        MLflow (for tracking experiments)
        Scikit-learn (for building and training models)
        Pandas and Numpy (for data manipulation)

    Packages:
        pandas
        numpy
        scikit-learn
        mlflow
### To install the required packages, run:

```pip install -r requirements.txt```

### File Structure

.
├── pipeline.py                  # Main Python script to run the ML pipeline <br>
├
├── requirements.txt             # Dependencies for the project <br>
├
├── python_env.yaml              # Python environment setup file for conda <br>
├
├── london_weather.csv           # The dataset used for training <br>
├
└── README.md                    # Project documentation <br>

### Workflow

    Data Loading and Preprocessing:
        The dataset (london_weather.csv) is loaded and cleaned by filling any missing values.
        The target variable, mean_temp, is separated from the features.

    Model Training:
        Two regression models are implemented:
            Linear Regression
            Random Forest Regressor
        The models are trained on the training data and evaluated using the testing data. The RMSE of each model is calculated.

    Model Evaluation and Logging:
        The performance of each model is tracked using MLflow. Key metrics, such as RMSE, are logged along with the model's parameters.
        The best-performing model is selected based on the RMSE.

    Output:
        A log of the RMSE for each model is printed to the console.
        MLflow logs model artifacts and metrics for future reference.

### How to Run the Project

    Prepare your environment:
        Install the dependencies listed in requirements.txt by running:

    pip install -r requirements.txt

### Run the pipeline:

    Execute the following command to run the ML pipeline:

    python pipeline.py

### View Results:

    After running the script, MLflow will log the model performance and parameters.
    You can view the experiments in the MLflow UI by running:

        mlflow ui

        Navigate to http://localhost:5000 to view the logged experiments.

### Files and Dependencies

    pipeline.py: Contains the main script that trains the models and logs the results using MLflow.
    requirements.txt: A list of Python dependencies required to run the project.
    python_env.yaml: A YAML file specifying the Python environment and dependencies for Conda.
    london_weather.csv: The dataset used to train the models. It contains historical weather data for London.

