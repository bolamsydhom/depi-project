# src/config.py

import mlflow
import torch
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec

# Define hyperparameters
HYPERPARAMETERS = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 100,
    "dropout_rate": 0.5,
    "weight_decay": 1e-4,
    "sgd_momentum": 0.9,
    "scheduler_gamma": 0.95,
    "pos_weight": 1.0,
}

# Define best parameters found after hyperparameter tuning
BEST_PARAMETERS = {
    "learning_rate": 0.0005,
    "batch_size": 64,
    "num_epochs": 80,
    "dropout_rate": 0.3,
    "weight_decay": 1e-5,
    "sgd_momentum": 0.9,
    "scheduler_gamma": 0.90,
    "pos_weight": 2.0,
}

# Example of defining a model signature
input_schema = Schema([
    ColSpec(type="tensor", name="input", shape=(None, 3, 224, 224)),  # Example input shape for an image
])

output_schema = Schema([
    ColSpec(type="tensor", name="output", shape=(None, 10)),  # Example output shape for 10 classes
])

SIGNATURE = ModelSignature(inputs=input_schema, outputs=output_schema)
