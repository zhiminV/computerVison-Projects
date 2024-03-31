# Zhimin Liang
# Spring 2024
# 5330 Project 5

"""
Purpose:

"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from DeepNetwork import MyNetwork  

# Load the trained model
model = MyNetwork()
model.load_state_dict(torch.load('DeepNetwork.pth'))
model.eval()

# Define transformations to preprocess the images
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load and preprocess your handwritten digit images
handwritten_digits = []
for i in range(10):
    image_path = f'handwritten_{i}.jpg' 
    image = Image.open(image_path)
    preprocessed_image = transform(image)
    handwritten_digits.append(preprocessed_image)

# Perform inference on the handwritten digit images
predicted_labels = []
for digit_image in handwritten_digits:
    with torch.no_grad():
        output = model(digit_image.unsqueeze(0))  # Add a batch dimension
        predicted_label = torch.argmax(output, dim=1).item()
        predicted_labels.append(predicted_label)

# Display the handwritten digit images along with their predicted labels
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(handwritten_digits[i].squeeze().numpy(), cmap='gray')
    ax.set_title(f'Predicted: {predicted_labels[i]}')
    ax.axis('off')

plt.tight_layout()
plt.show()
