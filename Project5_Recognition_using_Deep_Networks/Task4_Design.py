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
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import time


class MyCNN(nn.Module):
    def __init__(self, num_conv_layers, filter_size, num_filters):
        super(MyCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = 1

        for _ in range(num_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, num_filters, filter_size, padding=(filter_size - 1) // 2))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(2, 2))
            in_channels = num_filters

        # Dummy input to calculate the size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)
            for layer in self.conv_layers:
                dummy_input = layer(dummy_input)
            output_size = dummy_input.view(dummy_input.size(0), -1).size(1)

        self.fc = nn.Linear(output_size, 10)  # Dynamically set based on the output of conv layers

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Function to train the model
def train_model(model, train_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    losses = []

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Time: {time.time() - start_time} seconds")

    return losses

# Main function to execute the experiment
def main():
    # Define dimensions to evaluate
    num_conv_layers_list = [1, 2, 3]
    filter_size_list = [3, 5, 7]
    num_filters_list = [16, 32, 64]

    # Prepare FashionMNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Iterate over different combinations of dimensions
    for num_conv_layers in num_conv_layers_list:
        for filter_size in filter_size_list:
            for num_filters in num_filters_list:
                # Create and train the model
                model = MyCNN(num_conv_layers, filter_size, num_filters)
                print(f"\nTraining model with: Num Conv Layers={num_conv_layers}, Filter Size={filter_size}, Num Filters={num_filters}")
                losses = train_model(model, train_loader)

    print("Experiment completed.")

if __name__ == "__main__":
    main()


