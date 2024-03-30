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
import torch.optim as optim
import torchvision.transforms as transforms

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

    # Task 1_B Build a network model
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 20*4*4)  # Flatten the output from convolutions
        x = torch.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return torch.log_softmax(x, dim=1)
    

#task 1_C  Train the model
def train_model(model, train_loader, test_loader, num_epochs=5, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
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

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_losses[-1]:.4f}, '
              f'Train Acc: {train_accuracy[-1]*100:.2f}%, '
              f'Test Loss: {test_losses[-1]:.4f}, '
              f'Test Acc: {test_accuracy[-1]*100:.2f}%')

    return train_losses, test_losses, train_accuracy, test_accuracy

def plot_errors(train_losses, test_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.show()

def plot_accuracy(train_accuracy, test_accuracy):
    plt.plot(train_accuracy, label='Training Accuracy', color='blue')
    plt.plot(test_accuracy, label='Testing Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.show()


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

    # Task 1_A Get the MNIST digit data set
    mnist_test_set = get_mnist_test_set()
    plot_first_six_digits(mnist_test_set)

    #Task 1_C Train the model
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MyNetwork()
    train_losses, test_losses, train_accuracy, test_accuracy = train_model(model, train_loader, test_loader)
    plot_errors(train_losses, test_losses)
    plot_accuracy(train_accuracy, test_accuracy)

    #Task 1_D 

if __name__ == "__main__":
    main(sys.argv)
