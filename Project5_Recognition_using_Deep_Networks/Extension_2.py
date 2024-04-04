# Zhimin Liang
# Spring 2024
# 5330 Project 5

"""
Purpose:The purpose of this code is to perform real-time digit recognition using a pre-trained deep neural network model. 

"""
import cv2
import torch
import torchvision.transforms as transforms
from DeepNetwork import MyNetwork  # Import  trained network model

# Load pre-trained model
model = MyNetwork()
model.load_state_dict(torch.load('DeepNetwork.pth'))  
model.eval()  # Set the model to evaluation mode

# Define transformations for preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_frame(frame):
    '''
    Purpose: Preprocesses a frame to extract digits and bounding box coordinates.
    Parameters:
        - frame: A frame from the video feed (numpy array).
    Returns:
        - rois: List of region of interest (ROI) images containing digits.
        - bounding_boxes: List of tuples containing bounding box coordinates (x, y, width, height).
    '''
    # Apply preprocessing techniques to extract digits from the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of digits
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    bounding_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2500:  # Set minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y+h, x:x+w]
            rois.append(roi)
            bounding_boxes.append((x, y, w, h))

    return rois, bounding_boxes


def recognize_digits(frame):
    '''
    Purpose: Recognizes digits in a frame using a pre-trained model.
    Parameters:
        - frame: A frame from the video feed (numpy array).
    Returns:
        - recognized_digits: List of recognized digit labels.
        - bounding_boxes: List of tuples containing bounding box coordinates (x, y, width, height).
    '''
    rois, bounding_boxes = preprocess_frame(frame)

    # Convert each ROI to tensor and apply model
    recognized_digits = []
    for roi in rois:
        roi_tensor = transform(roi)
        with torch.no_grad():
            output = model(roi_tensor.unsqueeze(0))
            _, predicted = torch.max(output, 1)
            recognized_digits.append(predicted.item())

    return recognized_digits, bounding_boxes


def main():
    cap = cv2.VideoCapture(0)  

    while True:
        ret, frame = cap.read()  
        if not ret:
            break

        # Perform digit recognition on frame
        recognized_digits, bounding_boxes = recognize_digits(frame)

        # Overlay recognized digits and bounding boxes on frame
        for digit, (x, y, w, h) in zip(recognized_digits, bounding_boxes):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Digit Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
