# Zhimin Liang
# Spring 2024
# 5330 Project 5

"""
Purpose:

"""
import sys
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from DeepNetwork import MyNetwork 
 

def load_model(model_path='DeepNetwork.pth'):
    """
    Load the trained model from a file.
    """
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def evaluate_model(model, test_loader):
    """
    Evaluate the model on the first 10 examples from the test set.
    """
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if i == 10:  # Only process the first 10 examples
                break
            outputs = model(inputs)
            predicted = torch.max(outputs, 1)[1]
            print("\nExample", i + 1)
            print("Network output:", outputs[0].numpy().round(2))
            print("Predicted label:", predicted[i].item())
            print("Correct label:  ", labels[i].item())

def plot_predictions(model, test_loader):
    """
    Plot the first 9 digits of the test set with predictions.
    """
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if i >= 9:
                break
            outputs = model(inputs)
            predicted = torch.max(outputs, 1)[1]
            ax = axes[i // 3, i % 3]
            # Handle batch size of 1 correctly
            if inputs.shape[0] == 1:
                ax.imshow(inputs.squeeze(0).squeeze(), cmap='gray')  # Remove the extra batch dimension
            else:
                ax.imshow(inputs.squeeze(), cmap='gray')  # Squeeze only the channel dimension
            ax.set_title(f'Pred: {predicted[0].item()}')
            ax.axis('off')
    plt.show()



if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    model = load_model('DeepNetwork.pth')
    evaluate_model(model, test_loader)
    plot_predictions(model, torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False))
