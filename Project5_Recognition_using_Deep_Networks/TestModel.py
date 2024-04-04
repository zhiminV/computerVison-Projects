# Zhimin Liang
# Spring 2024
# 5330 Project 5

"""
Purpose:
This script is designed for evaluating and visualizing the performance of a trained deep neural network model on the MNIST test dataset. 
It loads a pre-trained model, evaluates its performance on the first 10 examples from the test set, and plots the predictions alongside the actual digits for the first 9 examples.

"""
import torch
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from DeepNetwork import MyNetwork

def load_model(model_path='DeepNetwork.pth'):
    """
    Load the trained model from a file.
    """
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate_model(model, test_loader):
    """
    Evaluate the model on the first 10 examples from the test set.

    Parameters:
    - model: The trained neural network model.
    - test_loader: DataLoader for the test dataset.

    Returns:
    None.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if i == 10:  # Process only the first 10 examples
                break
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            print(f"Example {i + 1}")
            print("Network output:", [round(float(val), 2) for val in outputs[0]])
            print("Predicted label:", predicted.item())
            print("Correct label:", labels.item())
            print()

def plot_predictions(model, test_loader):
    """
    Plot the first 9 digits of the test set with predictions.

    Parameters:
    - model: The trained neural network model.
    - test_loader: DataLoader for the test dataset.

    Returns:
    None.
    """
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if i >= 9:
                break
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            ax = axes[i // 3, i % 3]
            ax.imshow(inputs.squeeze().numpy(), cmap='gray')  
            ax.set_title(f'Pred: {predicted.item()}, Actual: {labels.item()}')
            ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = load_model('DeepNetwork.pth')
    evaluate_model(model, test_loader)
    plot_predictions(model, test_loader)