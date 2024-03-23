/*
Zhimin Liang
Spring 2024
CS5330 Project 4
Purpose: This program demonstrates the detection of Harris corners in a live video stream using OpenCV. Harris corners are key feature points in an image that can be used for various computer vision tasks such as image alignment, object detection, and tracking.

*/
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Open video capture device
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Failed to open video device." << endl;
        return -1;
    }

    Mat frame;
    Mat grayFrame;
    Mat corners;

    namedWindow("Harris Corners", WINDOW_NORMAL);

    while (true) {
        // Capture frame from video stream
        cap >> frame;
        if (frame.empty()) {
            cerr << "Failed to capture frame." << endl;
            break;
        }

        // Convert frame to grayscale
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // Detect Harris corners
        cornerHarris(grayFrame, corners, 2, 3, 0.04);

        // Normalize corner response
        normalize(corners, corners, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

        // Draw circles around detected corners
        for (int i = 0; i < corners.rows; ++i) {
            for (int j = 0; j < corners.cols; ++j) {
                if ((int)corners.at<float>(i, j) > 100) {
                    circle(frame, Point(j, i), 5, Scalar(0, 255, 0), 2);
                }
            }
        }

        // Show frame with detected corners
        imshow("Harris Corners", frame);

        // Check for exit key
        if (waitKey(1) == 'q') break;
    }

    // Release video capture and close windows
    cap.release();
    destroyAllWindows();

    return 0;
}
