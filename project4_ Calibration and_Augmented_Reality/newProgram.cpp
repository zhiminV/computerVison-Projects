/*
Zhimin Liang
Spring 2024
CS5330 Project 3

Purpose: This file contains the main program 
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include "calibrate.h"

using namespace cv;
using namespace std;

void calculateCameraPosition(const Mat& cameraMatrix, const Mat& distCoeffs, const vector<Point2f>& corners, const Size& patternSize, Mat& rvec, Mat& tvec) {
    // Define 3D object points for the calibration target
    vector<Point3f> objectPoints;
    for (int i = 0; i < patternSize.height; ++i) {
        for (int j = 0; j < patternSize.width; ++j) {
            objectPoints.push_back(Point3f(j, i, 0)); // Assuming the calibration target lies in the XY plane
        }
    }

    // Estimate the camera pose (rotation and translation)
    solvePnP(objectPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);

    // Print the rotation and translation vectors
    cout << "Rotation vector (rvec):\n" << rvec << endl;
    cout << "Translation vector (tvec):\n" << tvec << endl;
}


void draw3DAxes(cv::Mat& image, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs) {

    // Define the origin point
    cv::Point3f origin(0, 0, 0);

    // Define the endpoints of the axes
    std::vector<cv::Point3f> axesPoints;
    axesPoints.push_back(origin); // Origin point
    // X-axis endpoint
    axesPoints.push_back(cv::Point3f(-40, 0, 0)); 
    // Y-axis endpoint
    axesPoints.push_back(cv::Point3f(0, 40, 0)); 
    // Z-axis endpoint
    axesPoints.push_back(cv::Point3f(0, 0, 40)); 

    // Project the axes points onto the image plane
    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(axesPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

    // Draw the axes lines
    cv::line(image, projectedPoints[0], projectedPoints[1], cv::Scalar(0, 0, 255), 5); // X-axis (red)
    cv::line(image, projectedPoints[0], projectedPoints[2], cv::Scalar(0, 255, 0), 5); // Y-axis (green)
    cv::line(image, projectedPoints[0], projectedPoints[3], cv::Scalar(255, 0, 0), 5); // Z-axis (blue)
}


int main(int argc, char *argv[]) {
    // Load camera calibration parameters from file
    cv::Mat cameraMatrix, distCoeffs;
    cv::FileStorage fs("calibration.csv", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open calibration file!" << endl;
        return -1;
    }
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    // Open the video device
    cv::VideoCapture capdev(0);
    if (!capdev.isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }

    // Get the size of the captured frames
    cv::Size frameSize(capdev.get(cv::CAP_PROP_FRAME_WIDTH), capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Frame size: %d x %d\n", frameSize.width, frameSize.height);

    // Create a window for displaying the captured frames
    cv::namedWindow("Detect Corners", 1);

    std::vector<cv::Point2f> corners;
    cv::Size patternSize(9, 6); // Change this to match your pattern size

    cv::Mat rvec, tvec; // Declare rotation and translation vectors

    while (true) {
        // Capture a new frame from the video device
        cv::Mat frame;
        capdev >> frame;

        // Check if the frame is empty
        if (frame.empty()) {
            printf("Frame is empty\n");
            break;
        }

        // Detect and extract target corners
        DetectAndExtractTargetCorners(frame, corners, patternSize);
        // cv::imshow("Detect ", frame);

          // Calculate camera pose and draw 3D axes
        if (!corners.empty()) {
            calculateCameraPosition(cameraMatrix, distCoeffs, corners, patternSize, rvec, tvec);
            draw3DAxes(frame, rvec, tvec, cameraMatrix, distCoeffs);
            cv::imshow("draw3DAxes ", frame);
        }



        // Press 'q' to quit the loop
        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        }
        // Press 'c' to calculate camera pose and draw 3D axes
        else if (key == 'c') {
            // Call the function to calculate camera pose
            calculateCameraPosition(cameraMatrix, distCoeffs, corners, patternSize, rvec, tvec);
        }
       
    }

    // Release the video device and close the window
    capdev.release();
    cv::destroyAllWindows();

    return 0;
}
