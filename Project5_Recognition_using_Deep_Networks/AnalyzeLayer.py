# Zhimin Liang
# Spring 2024
# 5330 Project 5

"""
Purpose:

"""

import torch
import matplotlib.pyplot as plt
from DeepNetwork import MyNetwork
import cv2
import torchvision.transforms as transforms
from torchvision import datasets 

def load_model(model_path='DeepNetwork.pth'):
    """Load the trained model."""
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    return model

def examine_network(model):
    """Print the model to examine its structure and layer names."""
    print(model)

def analyze_first_layer(model):
    """Get the weights of the first layer (conv1) and visualize the ten filters."""
    weights = model.conv1.weight
    print("Shape of weights in conv1 layer:", weights.shape)
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(weights[i, 0].detach().numpy(), cmap='gray')
        ax.set_title(f'Filter {i+1}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

def show_filter_effects(model, train_dataset):
    """Show the effect of the filters applied to the first training example image."""
    first_image, _ = train_dataset[0]  # Get the first training example image
    first_image = first_image.unsqueeze(0)  # Add batch dimension
    weights = model.conv1.weight

    filtered_images = []
    with torch.no_grad():
        for i in range(10):
            filter_weights = weights[i, 0].unsqueeze(0).unsqueeze(0)
            filtered_image = cv2.filter2D(first_image.squeeze().numpy(), -1, filter_weights.squeeze().numpy())
            filtered_images.append(filtered_image)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(filtered_images[i], cmap='gray')
        ax.set_title(f'Filtered Image {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('conv1_filterResults.png')  
    plt.show()

def main():
    
   # Import the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    
    # Load the trained model
    model = load_model()

    # Task 2: Examine the network
    examine_network(model)

    # Task 2_A: Analyze the first layer
    analyze_first_layer(model)

    # Task 2_B: Show the effect of the filters
    show_filter_effects(model, train_dataset)

if __name__ == "__main__":
    main()
