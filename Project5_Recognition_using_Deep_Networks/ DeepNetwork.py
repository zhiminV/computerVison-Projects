# Zhimin Liang
# Spring 2024
# 5330 Project 5

# Purpose:

# Your name here and a short header

# import statements
import sys
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# class definitions
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = nn.Dropout(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=20*4*4, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 20*4*4)  # Flatten the output from convolutions
        x = torch.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return torch.log_softmax(x, dim=1)

# useful functions with a comment for each function
def get_mnist_test_set():
    """
    Load the MNIST test set from torchvision.datasets.MNIST.

    Returns:
    - mnist_test: MNIST test dataset
    """
    mnist_test = datasets.MNIST(root='./data', train=False, download=True)
    return mnist_test

def plot_first_six_digits(test_set):
    """
    Plot the first six example digits from the test set.

    Args:
    - test_set: MNIST test dataset
    """
    # Extract the first six images and their labels
    images = test_set.data[:6]
    labels = test_set.targets[:6]

    # Plot the first six example digits
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def main(argv):
    """
    Main function to execute the code.

    Args:
    - argv: command line arguments
    """
    # handle any command line arguments in argv

    # Load the MNIST test set
    mnist_test_set = get_mnist_test_set()

    # Plot the first six example digits
    plot_first_six_digits(mnist_test_set)

if __name__ == "__main__":
    main(sys.argv)
