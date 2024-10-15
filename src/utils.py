import mlflow

def log_results(probabilities):
    with mlflow.start_run() as run:
        # Log the model and parameters
        mlflow.log_param("model", "ResNet18")  # Change according to your model
        mlflow.log_param("num_classes", len(probabilities))
        
        # Log class probabilities
        for i, prob in enumerate(probabilities):
            mlflow.log_metric(f"Probability_class_{i}", prob)
        
        # Optionally, log additional metrics (if you have ground truth labels)
        # For example:
        # mlflow.log_metric("accuracy", accuracy_value)
        # mlflow.log_metric("f1_score", f1_score_value)

        # If you're saving a model, you can log it
        # mlflow.pytorch.log_model(model, "model")

        print("Results logged to MLflow.")

