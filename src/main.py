# src/main.py
import os
import torch
import mlflow
import mlflow.pytorch
import torchvision.transforms as transforms
from PIL import Image
from RAG_GGUF import RAG_Chain as MyModel  # Your custom model
from utils import log_results  # Utility function to log metrics

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
model = MyModel().to(device)
model.eval()

# Load and preprocess an image
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

# Inference function
def infer(image):
    with torch.no_grad():
        output = model(image)
        return torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()

# Main function
def main(image_path):
    with mlflow.start_run() as run:
        image = load_image(image_path)
        probabilities = infer(image)
        log_results(probabilities)

if __name__ == "__main__":
    image_path = "data/images/sample.jpg"  # Update with your image path
    main(image_path)
