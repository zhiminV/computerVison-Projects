/*
Zhimin Liang
Spring 2024
CS5330 Project 4

Purpose: This program demonstrates the use of camera calibration parameters to estimate the pose of a calibration target in a video stream and project 3D axes onto the image plane to visualize the orientation of the target.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

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

        // Detect checkerboard corners
        Size patternSize(9, 6); // Assuming a 9x6 checkerboard pattern
        vector<Point2f> corners;
        bool patternFound = findChessboardCorners(grayImage, patternSize, corners,
                                                   CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

    // Define 3D object points for the calibration target
        vector<Point3f> objectPoints;
        for (int i = 0; i < patternSize.height; ++i) {
            for (int j = 0; j < patternSize.width; ++j) {
                objectPoints.push_back(Point3f(j, i, 0)); // Assuming the calibration target lies in the XY plane
            }
        }
        if (patternFound) {
            // Refine corner positions
            cornerSubPix(grayImage, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

            // Solve PnP to estimate pose
            Mat rvec, tvec;
            solvePnP(objectPoints, corners, cameraMatrix, distortionCoefficients, rvec, tvec);

            // Print rotation and translation data
            cout << "Rotation vector:" << endl << rvec << endl;
            cout << "Translation vector:" << endl << tvec << endl;

            // Project 3D axes onto image plane
            vector<Point3f> axisPoints;
            axisPoints.push_back(Point3f(0, 0, 0));  // Origin
            axisPoints.push_back(Point3f(3, 0, 0));  // X-axis
            axisPoints.push_back(Point3f(0, 3, 0));  // Y-axis
            axisPoints.push_back(Point3f(0, 0, -3)); // Z-axis

            vector<Point2f> projectedPoints;
            projectPoints(axisPoints, rvec, tvec, cameraMatrix, distortionCoefficients, projectedPoints);

            // Draw axes
            arrowedLine(frame, projectedPoints[0], projectedPoints[1], Scalar(0, 0, 255), 2); // X-axis (red)
            arrowedLine(frame, projectedPoints[0], projectedPoints[2], Scalar(0, 255, 0), 2); // Y-axis (green)
            arrowedLine(frame, projectedPoints[0], projectedPoints[3], Scalar(255, 0, 0), 2); // Z-axis (blue)
        }

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
