#%% imports
import torch
import mlflow
import mlflow.pytorch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Specify MLflow tracking server
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5001')
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Load pre-trained model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model = model.to(device)
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load and preprocess an image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and send to device
    return image

# Function to perform inference
def infer(image):
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities.cpu().numpy()

# Function to log results to MLflow
def log_results(probabilities):
    with mlflow.start_run() as run:
        mlflow.log_param("model", "ResNet18")
        mlflow.log_param("num_classes", len(probabilities))
        
        # Log probabilities as a metric
        for i, prob in enumerate(probabilities):
            mlflow.log_metric(f"Probability_class_{i}", prob)

# Main function to run the pipeline
def main(image_path):
    # Load and preprocess the image
    image = load_image(image_path)
    
    # Perform inference
    probabilities = infer(image)
    
    # Log results
    log_results(probabilities)
    
    # Print probabilities
    print("Class probabilities:", probabilities)

# Example usage
if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Update this path to your image
    main(image_path)

