# Zhimin Liang
# Spring 2024
# 5330 Project 5

"""
Purpose:

"""

import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
from DeepNetwork import MyNetwork 

class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.resize(x, [133, 133])
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

def load_pretrained_model(model_path='DeepNetwork.pth'):
    """Load the trained model from a file."""
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    return model

def freeze_weights(model):
    """Freeze the network weights."""
    for param in model.parameters():
        param.requires_grad = False

def replace_last_layer(model):
    """Replace the last layer with a new Linear layer with three nodes."""
    model.fc2 = nn.Linear(50, 3)  # Replace the last layer with three nodes
    return model


def train_model(model, train_loader, epochs=10):
    """Train the modified network on the Greek letter dataset."""
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {accuracy}%")
        losses.append(epoch_loss)

    return losses


def plot_training_loss(losses):
    """Plot the training loss over epochs."""
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss')
    plt.show()


def main():
    # Step 1: Load pre-trained model
    model = load_pretrained_model()
    
    # Step 2: Freeze network weights
    freeze_weights(model)  
    
    # Step 3: Replace last layer
    model = replace_last_layer(model)  

    # Step 4: Define transformation for the Greek letter dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Step 5: directory containing the Greek letter dataset
    training_set_path = 'greek_train'

    # Step 6: Create DataLoader for the Greek letter dataset
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(training_set_path, transform=transform),
        batch_size=5,
        shuffle=True
    )

    # Step 7: Train the modified network on Greek letter dataset
    losses = train_model(model, greek_train, epochs=10)

    # Step 8: Plot the training loss
    plot_training_loss(losses)

    # Step 9: Print out the modified network
    print("Modified network structure:")
    print(model)


if __name__ == "__main__":
    main()
