/*
Zhimin Liang
Spring 2024
CS5330 Project 3
Purpose: Implementation of Task 6 - Creating a Virtual Object (Cylinder)
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Function to draw a cylinder in 3D space
void drawCylinder(Mat& frame, const Mat& cameraMatrix, const Mat& distortionCoefficients,
                  const Mat& rvec, const Mat& tvec, double radius, double height,
                  const Scalar& color = Scalar(255, 0, 0), int thickness = 2) {
    // Define points for the base circle
    vector<Point3f> baseCirclePoints;
    int sides = 50; // Number of sides of the cylinder (higher value for smoother circle)
    for (int i = 0; i < sides; ++i) {
        float angle = (float)i * 2 * CV_PI / sides;
        float x = radius * cos(angle);
        float y = radius * sin(angle);
        baseCirclePoints.push_back(Point3f(x, y, 0));
    }

    // Project base circle points onto the image plane
    vector<Point2f> projectedPoints;
    projectPoints(baseCirclePoints, rvec, tvec, cameraMatrix, distortionCoefficients, projectedPoints);

    // Draw base circle
    for (int i = 0; i < sides - 1; ++i) {
        line(frame, projectedPoints[i], projectedPoints[i + 1], color, thickness);
    }
    line(frame, projectedPoints[sides - 1], projectedPoints[0], color, thickness);

    // Project top circle points by translating along z-axis by height
    vector<Point3f> topCirclePoints;
    for (const auto& point : baseCirclePoints) {
        topCirclePoints.push_back(Point3f(point.x, point.y, height));
    }
    vector<Point2f> projectedTopPoints;
    projectPoints(topCirclePoints, rvec, tvec, cameraMatrix, distortionCoefficients, projectedTopPoints);

    // Draw top circle
    for (int i = 0; i < sides - 1; ++i) {
        line(frame, projectedPoints[i], projectedTopPoints[i], color, thickness);
    }
    line(frame, projectedPoints[sides - 1], projectedTopPoints[sides - 1], color, thickness);

    // Draw vertical lines connecting base and top circles
    for (int i = 0; i < sides; ++i) {
        line(frame, projectedPoints[i], projectedTopPoints[i], color, thickness);
    }
}

int main() {
    // Read camera calibration parameters from file
    FileStorage fs("calibration.csv", FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open calibration file." << endl;
        return -1;
    }

    Mat cameraMatrix, distortionCoefficients;
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distortionCoefficients;
    fs.release();

    // Start video capture
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Failed to open video device." << endl;
        return -1;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "Failed to capture frame." << endl;
            break;
        }

        // Convert frame to grayscale
        Mat grayImage;
        cvtColor(frame, grayImage, COLOR_BGR2GRAY);

        // Dummy values for rotation and translation vectors (not used here)
        Mat rvec = Mat::zeros(3, 1, CV_64F);
        Mat tvec = Mat::zeros(3, 1, CV_64F);

        // Draw a cylinder
        drawCylinder(frame, cameraMatrix, distortionCoefficients, rvec, tvec, 50, 100);

        // Show frame
        imshow("Frame", frame);

        // Exit loop if 'q' is pressed
        if (waitKey(1) == 'q') break;
    }

    // Release video capture and close windows
    cap.release();
    destroyAllWindows();

    return 0;
}
