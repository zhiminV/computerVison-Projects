# Zhimin Liang
# Spring 2024
# 5330 Project 5

"""
Purpose:

"""

import torch
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
from torchvision import datasets, models
from torchvision import transforms
import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


def load_pretrained_model():
    """Load a pre-trained VGG16 model."""
    model = models.vgg16(pretrained=True)
    return model

def examine_network(model):
    """Print the model to examine its structure and layer names."""
    print(model)

def analyze_first_layer(model):
    """Get the weights of the first convolutional layer and visualize the filters."""
    # VGG16's first conv layer is named features[0]
    weights = model.features[0].weight.data
    print("Shape of weights in the first conv layer:", weights.shape)
    
    # Visualize the first 10 filters
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        # VGG16 filters are 3-channel (RGB), so we take the mean to visualize them in grayscale
        filter_avg = weights[i].mean(0)
        ax.imshow(filter_avg, cmap='gray')
        ax.set_title(f'Filter {i+1}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

def show_filter_effects(model, train_dataset):
    first_image, _ = train_dataset[0]  # Get the first training example image
    first_image_rgb = first_image.unsqueeze(0)  # Ensure it's in the shape (B, C, H, W)

    # Get the weights of the first conv layer; no need to repeat or unsqueeze
    weights = model.features[0].weight.data

    # Applying filters individually
    with torch.no_grad():
        filtered_images = []
        for i in range(10):  # Apply the first 10 filters
            filter_weights = weights[i].unsqueeze(1)  
            filtered_image = torch.nn.functional.conv2d(first_image_rgb, filter_weights, groups=3, padding=1)
            # Mean across the channels to get a single-channel image
            filtered_image = filtered_image.mean(1, keepdim=True)
            filtered_images.append(filtered_image[0])


        fig, axes = plt.subplots(2, 5, figsize=(20, 8)) 
        for i, ax in enumerate(axes.flat):
            # Convert the tensor to a numpy array and squeeze out any extra dimensions
            img = filtered_images[i].squeeze().numpy()

            # The cmap='gray' is used to display the image in grayscale
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')  # Hide the axes ticks

        plt.tight_layout()  
        plt.show()  



def main():
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Assuming you want to resize to 224x224
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)


    model = load_pretrained_model()

    examine_network(model)
    analyze_first_layer(model)
    show_filter_effects(model, train_dataset)

if __name__ == "__main__":
    main()
