# Zhimin Liang
# Spring 2024
# 5330 Project 5

"""
Purpose:
This script is designed to implement and train a deep neural network model using the MNIST dataset.
The project aims to accurately recognize and classify handwritten digits through the application
of convolutional neural networks (CNNs). It covers data preprocessing, model architecture design,
training, and evaluation. Additionally, it provides functionality to save the trained model.
"""

# Import statements
import sys
import random
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import cv2  
import numpy as np 

# Define the network class
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = nn.Dropout(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return nn.functional.log_softmax(x, dim=1)

# Function to generate Gabor filter bank
def gabor_filter_bank(size=28, num_orientations=8, num_scales=5):
    filters = []
    for theta in range(num_orientations):
        theta = theta / num_orientations * np.pi
        for sigma in (1, 3):
            for freq in (0.1, 0.2):
                kernel = cv2.getGaborKernel((size, size), sigma, theta, freq, 0.5, 0, ktype=cv2.CV_32F)
                filters.append(kernel)
    return filters

# Replace first layer with Gabor filter bank
def replace_first_layer(model, filter_bank):
    weights = torch.tensor(np.array(filter_bank)).unsqueeze(1)
    num_output_channels = 10  
    model.conv1 = nn.Conv2d(in_channels=1, out_channels=num_output_channels, kernel_size=3, stride=1, padding=1, bias=False)
    with torch.no_grad():
        model.conv1.weight = nn.Parameter(weights)

# Retrain network while holding first layer constant
def retrain_network(model, train_loader, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Set the first layer to be non-trainable
    for param in model.conv1.parameters():
        param.requires_grad = False

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Main function
def main(argv):
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load MNIST data
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create network model
    model = MyNetwork()

    # Generate Gabor filter bank
    gabor_filters = gabor_filter_bank()  

    # Replace first layer with Gabor filter bank
    replace_first_layer(model, gabor_filters)

    # Retrain network while holding first layer constant
    retrain_network(model, train_loader, num_epochs=5)  

if __name__ == "__main__":
    main(sys.argv)
