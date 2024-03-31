# Zhimin Liang
# Spring 2024
# 5330 Project 5

"""
Purpose:

"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from DeepNetwork import MyNetwork  

def load_model(model_path='DeepNetwork.pth'):
    """Load the trained model."""
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_images(image_paths):
    """Preprocess the handwritten digit images."""
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    handwritten_digits = []
    for path in image_paths:
        image = Image.open(path)
        preprocessed_image = transform(image)
        handwritten_digits.append(preprocessed_image)
    return handwritten_digits

def perform_inference(model, handwritten_digits):
    """Perform inference on the handwritten digit images."""
    predicted_labels = []
    with torch.no_grad():
        for digit_image in handwritten_digits:
            output = model(digit_image.unsqueeze(0))  # Add a batch dimension
            predicted_label = torch.argmax(output, dim=1).item()
            predicted_labels.append(predicted_label)
    return predicted_labels

def display_results(handwritten_digits, predicted_labels):
    """Display the handwritten digit images along with their predicted labels."""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(handwritten_digits[i].squeeze().numpy(), cmap='gray')
        ax.set_title(f'Predicted: {predicted_labels[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Load the trained model
    model = load_model('DeepNetwork.pth')

    # Define image paths
    image_paths = [f'resized_digit{i}.jpg' for i in range(10)]

    # Preprocess the images
    handwritten_digits = preprocess_images(image_paths)

    # Perform inference on the images
    predicted_labels = perform_inference(model, handwritten_digits)

    # Display the results
    display_results(handwritten_digits, predicted_labels)

if __name__ == "__main__":
    main()
