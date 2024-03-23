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

// Function to draw a cube given its vertices
void drawCube(Mat& frame, const vector<Point2f>& projectedPoints) {
    // Connect the vertices to form the cube
    // Front face
    line(frame, projectedPoints[0], projectedPoints[1], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[1], projectedPoints[2], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[2], projectedPoints[3], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[3], projectedPoints[0], Scalar(0, 0, 255), 2);
    // Back face
    line(frame, projectedPoints[4], projectedPoints[5], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[5], projectedPoints[6], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[6], projectedPoints[7], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[7], projectedPoints[4], Scalar(0, 0, 255), 2);
    // Connect front and back faces
    line(frame, projectedPoints[0], projectedPoints[4], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[1], projectedPoints[5], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[2], projectedPoints[6], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[3], projectedPoints[7], Scalar(0, 0, 255), 2);
}

void drawPyramid(Mat& frame, const Mat& rvec, const Mat& tvec, const Mat& cameraMatrix, const Mat& distortionCoefficients) {
    // Define the vertices of the pyramid
    vector<Point3f> pyramidPoints;
    pyramidPoints.push_back(Point3f(0, 0, 0)); // Base center
    pyramidPoints.push_back(Point3f(-1, -1, 0)); // Base vertices
    pyramidPoints.push_back(Point3f(1, -1, 0));
    pyramidPoints.push_back(Point3f(1, 1, 0));
    pyramidPoints.push_back(Point3f(-1, 1, 0));
    pyramidPoints.push_back(Point3f(0, 0, 2)); // Apex

    // Project pyramid points onto image plane
    vector<Point2f> projectedPoints;
    projectPoints(pyramidPoints, rvec, tvec, cameraMatrix, distortionCoefficients, projectedPoints);

    // Draw lines to form the pyramid
    line(frame, projectedPoints[1], projectedPoints[2], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[2], projectedPoints[3], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[3], projectedPoints[4], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[4], projectedPoints[1], Scalar(0, 0, 255), 2);

    line(frame, projectedPoints[0], projectedPoints[1], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[0], projectedPoints[2], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[0], projectedPoints[3], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[0], projectedPoints[4], Scalar(0, 0, 255), 2);

    line(frame, projectedPoints[5], projectedPoints[1], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[5], projectedPoints[2], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[5], projectedPoints[3], Scalar(0, 0, 255), 2);
    line(frame, projectedPoints[5], projectedPoints[4], Scalar(0, 0, 255), 2);
}
    
// Function to draw a cylinder given its vertices
void drawCylinder(Mat& frame, const vector<Point2f>& projectedPoints) {
    // Draw the base of the cylinder
    circle(frame, projectedPoints[0], 5, Scalar(0, 255, 0), FILLED); // Center of the base
    circle(frame, projectedPoints[1], 5, Scalar(0, 255, 0), FILLED); // Point on the circumference

    // Draw lines connecting the base to the top of the cylinder
    line(frame, projectedPoints[0], projectedPoints[2], Scalar(0, 255, 0), 2);
    line(frame, projectedPoints[1], projectedPoints[3], Scalar(0, 255, 0), 2);
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

            // Project 3D cube onto image plane
            vector<Point3f> cubePoints = {
                Point3f(-1, -1, -1), Point3f(1, -1, -1), Point3f(1, 1, -1), Point3f(-1, 1, -1),
                Point3f(-1, -1, 1), Point3f(1, -1, 1), Point3f(1, 1, 1), Point3f(-1, 1, 1)
            };
            vector<Point2f> projectedCubePoints;
            projectPoints(cubePoints, rvec, tvec, cameraMatrix, distortionCoefficients, projectedCubePoints);
            drawCube(frame, projectedCubePoints);

            // Project 3D cylinder onto image plane
            vector<Point3f> cylinderPoints = {
                Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 0, 2), Point3f(1, 0, 2)
            };
            vector<Point2f> projectedCylinderPoints;
            projectPoints(cylinderPoints, rvec, tvec, cameraMatrix, distortionCoefficients, projectedCylinderPoints);
            drawCylinder(frame, projectedCylinderPoints);

             // Project 3D pyramid onto image plane
             drawPyramid(frame, rvec, tvec, cameraMatrix, distortionCoefficients);

        }


        // Show frame
        imshow("Frame", frame);

        // Exit loop if 'q' is pressed
        if (waitKey(1) == 'q') {
            break;
        }
    }

    // Release video capture and close windows
    cap.release();
    destroyAllWindows();
    
    return 0;
}
