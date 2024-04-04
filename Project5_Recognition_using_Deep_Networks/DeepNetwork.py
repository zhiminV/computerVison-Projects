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

# import statements
import sys
import random
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Class definitions
class MyNetwork(nn.Module):
    """
    Purpose:
    Defines the architecture of the neural network model.

    Description:
    This class defines a convolutional neural network (CNN) model for digit recognition. It consists
    of two convolutional layers with max-pooling, followed by two fully connected layers. ReLU activation
    functions are applied after each convolutional layer, and a dropout layer is added before the second
    fully connected layer. The output layer produces log probabilities using a softmax function.
    """
     
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = nn.Dropout(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return nn.functional.log_softmax(x, dim=1)

# Training function with improvements
def train_model(model, train_loader, test_loader, num_epochs=5, learning_rate=0.001):
    """
    Purpose:
    Trains the neural network model.

    Parameters:
    - model: The neural network model.
    - train_loader: DataLoader for the training dataset.
    - test_loader: DataLoader for the testing dataset.
    - num_epochs: Number of training epochs (default: 5).
    - learning_rate: Learning rate for the optimizer (default: 0.001).

    Returns:
    Four lists containing training losses, testing losses, training accuracies, and testing accuracies
    for each epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        correct_train = 0
        total_train = 0
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs + torch.randn_like(inputs) * 0.1  # Adding random noise for data augmentation
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        train_accuracy.append(correct_train / total_train)

        # Testing
        model.eval()
        correct_test = 0
        total_test = 0
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                running_loss += loss.item()
            test_losses.append(running_loss / len(test_loader))
            test_accuracy.append(correct_test / total_test)
        
        scheduler.step()  # Adjust learning rate
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_losses[-1]:.4f}, '
              f'Train Acc: {train_accuracy[-1]*100:.2f}%, '
              f'Test Loss: {test_losses[-1]:.4f}, '
              f'Test Acc: {test_accuracy[-1]*100:.2f}%')

    return train_losses, test_losses, train_accuracy, test_accuracy

def plot_errors(train_losses, test_losses):
    """
    Purpose:
    Plots the training and testing losses.

    Parameters:
    - train_losses: List of training losses for each epoch.
    - test_losses: List of testing losses for each epoch.

    Returns:
    None.
    """
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.show()

def plot_accuracy(train_accuracy, test_accuracy):
    """
    Purpose:
    Plots the training and testing accuracies.

    Parameters:
    - train_accuracy: List of training accuracies for each epoch.
    - test_accuracy: List of testing accuracies for each epoch.

    Returns:
    None.
    """
    plt.plot(train_accuracy, label='Training Accuracy', color='blue')
    plt.plot(test_accuracy, label='Testing Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.show()

def get_mnist_test_set():
    """
    Purpose:
    Retrieves the MNIST test dataset.

    Parameters:
    None.

    Returns:
    The MNIST test dataset.
    """
    mnist_test = datasets.MNIST(root='./data', train=False, download=True)
    return mnist_test

def plot_random_six_digits(test_set):
    """
    Purpose:
    Plots six random digits from the test dataset.

    Parameters:
    - test_set: The MNIST test dataset.

    Returns:
    None.
    """
    indices = random.sample(range(len(test_set)), 6)
    images = [test_set[i][0] for i in indices]
    labels = [test_set[i][1] for i in indices]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

 
# Saves the trained model to a file.
def save_model(model, file_path='model.pth'):
    torch.save(model.state_dict(), file_path)

def main(argv):
    mnist_test_set = get_mnist_test_set()
    plot_random_six_digits(mnist_test_set)

    transform = transforms.Compose([
        transforms.RandomRotation(10),  # Randomly rotate images by 10 degrees
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MyNetwork()
    train_losses, test_losses, train_accuracy, test_accuracy = train_model(model, train_loader, test_loader)
    plot_errors(train_losses, test_losses)
    plot_accuracy(train_accuracy, test_accuracy)

    save_model(model, 'DeepNetwork.pth')
    print("Model saved successfully.")

if __name__ == "__main__":
    main(sys.argv)
