/*
Zhimin Liang
Spring 2024
CS5330 Project 1

Purpose: This file contains the main program for displaying live video from a camera device.
*/

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <sys/time.h>
#include "filter.h"
#include "faceDetect.h"

/*
Main function for image processing using OpenCV.

This program captures video from a camera device, processes each frame based on the last key pressed,
and displays the resulting images. Various filters and effects can be applied to the video stream.

Arguments:
  int argc - Number of command-line arguments.
  char *argv[] - Array of command-line arguments.

Returns:
  0 on successful execution.
*/
int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;

    // open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return (-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1);
    cv::Mat frame, sobelX, sobelY, gradientMagnitude;

    char last_keypress = '\0';

    for (;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        // Check the last key pressed and modify the image accordingly
        if (last_keypress == 'g') {
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
            cv::imshow("greyscale", frame);
        } else if (last_keypress == 'h') {
            greyscale(frame, frame); // Apply the custom greyscale transformation
            cv::imshow("custom_greyscale", frame);
        } else if (last_keypress == 'b') {
            blur5x5_2(frame, frame); // Apply the blurred version of the video stream (in color)
            cv::imshow("blurred", frame);
        } else if (last_keypress == 't') {
            sepiaTone(frame, frame);
            cv::imshow("sepia", frame); // Apply the a sepia tone filter without vignetting
        } else if (last_keypress == 'v') {
            sepiawithVignette(frame, frame);
            cv::imshow("sepia vignetting", frame); // Apply the a sepia tone filter with vignetting
        } else if (last_keypress == 'x') {
            sobelX3x3(frame, sobelX); // Apply Sobel X filter
            cv::convertScaleAbs(sobelX, sobelX);
            cv::imshow("Sobel X", sobelX);
        } else if (last_keypress == 'y') {
            sobelY3x3(frame, sobelY); // Apply Sobel Y filter
            cv::convertScaleAbs(sobelY, sobelY);
            cv::imshow("Sobel Y", sobelY);
        } else if (last_keypress == 'm') {
            sobelX3x3(frame, sobelX);
            sobelY3x3(frame, sobelY);

            magnitude(sobelX, sobelY, gradientMagnitude); // Compute gradient magnitude

            // Convert to 8-bit for visualization
            cv::convertScaleAbs(gradientMagnitude, gradientMagnitude);

            cv::imshow("Gradient", gradientMagnitude);
        } else if (last_keypress == 'l') {
            blurQuantize(frame, frame, 10); // You can adjust the number of levels as needed
            cv::imshow("Blur Quantize", frame);
        } else if (last_keypress == 'f') {
            cv::Mat grey;
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);

            // Detect faces
            std::vector<cv::Rect> faces;
            detectFaces(grey, faces);

            // Draw rectangles around detected faces
            drawBoxes(frame, faces);

            cv::imshow("Face Detection", frame);

        } else if (last_keypress == 'o') {
            blurOutsideFaces(frame, frame);
            cv::imshow("Blur Outside Faces", frame);
        } else if (last_keypress == 'e') {
            embossingEffect(frame, frame);
            cv::imshow("Embossing Effect", frame);
        } else if (last_keypress == 'k') {
            cv::Mat grayImage;
            cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);

            std::vector<cv::Rect> faces;
            detectFaces(grayImage, faces);
            drawBoxes(frame, faces);
           
            addSparklesAboveFaces(frame, faces);
            cv::imshow("Sparkles Faces", frame);
        }else if (last_keypress == 'n') {
            cartoonize(frame,frame);
            cv::imshow("cartoonize Faces", frame);
        }
        cv::imshow("Video", frame);

        // see if there is a waiting keystroke
        char key = cv::waitKey(10);

        if (key == 'q') {
            break;
        } else if (key == 's') {
            cv::imwrite("captured_frame.jpg", frame);
            printf("Frame saved to captured_frame.jpg\n");
        } else {
            last_keypress = key;
        }
    }

    delete capdev;
    return (0);
}
