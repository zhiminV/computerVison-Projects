/*
Zhimin Liang
Spring 2024
CS5330 Project 4

Purpose: This file contains functions related to camera calibration for a computer vision project. The purpose of this file is to provide a set of functions for detecting and extracting corner points from a calibration target, saving the detected corner points along with corresponding 3D world points for calibration, and saving the intrinsic camera parameters (camera matrix and distortion coefficients) to a file. These functions are essential steps in the camera calibration process, which is crucial for accurate computer vision tasks such as object tracking, augmented reality, and 3D reconstruction. By encapsulating these functionalities in a separate file, the codebase becomes modular, maintainable, and reusable across different projects or modules within the same project.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "calibrate.h"
#include <fstream>

using namespace cv;
using namespace std;

std::vector<std::vector<cv::Point2f>> corner_list;
std::vector<std::vector<cv::Vec3f>> point_list;

/*
Function: DetectAndExtractTargetCorners
Description: This function detects and extracts the corners of a calibration target (e.g., a checkerboard pattern) in the input frame. It converts the frame to grayscale, performs corner detection using the findChessboardCorners function from OpenCV, refines the corner positions using cornerSubPix, and draws the detected corners on the frame if found.

Parameters:
- frame: Input frame (image) in which corners are to be detected.
- corners: Detected corner points will be stored in this vector.
- patternSize: Size of the calibration target pattern (e.g., number of inner corners).

Returns: None
*/
void DetectAndExtractTargetCorners(cv::Mat& frame, std::vector<cv::Point2f>& corners,const Size& patternSize) {
    Mat grayImage;
    cvtColor(frame, grayImage, COLOR_BGR2GRAY);

    bool patternFound = findChessboardCorners(grayImage, patternSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

    if (patternFound) {
        cornerSubPix(grayImage, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
        drawChessboardCorners(frame, patternSize, Mat(corners), patternFound);
    }
}


/*
Function: saveCalibrationData
Description: This function saves the detected corner locations and the corresponding 3D world points in vectors for later use in camera calibration. It takes the detected corners and the pattern size as input and constructs the 3D world point set assuming unit squares.

Parameters:
- corners: Detected corner points of the calibration target.
- patternSize: Size of the calibration target pattern (e.g., number of inner corners).

Returns: None
*/

void saveCalibrationData(const vector<Point2f>& corners, const Size& patternSize) {
    // Save the corner locations
    corner_list.push_back(corners);

    // Create and save the 3D world point set
    std::vector<cv::Vec3f> point_set;
    for (int i = 0; i < patternSize.height; ++i) {
        for (int j = 0; j < patternSize.width; ++j) {
            point_set.push_back(cv::Vec3f(j, i, 0)); // Assuming squares are 1 unit apart in the world
        }
    }
    point_list.push_back(point_set);
}

//Task 3: Enable the user to write out the intrinsic parameters to a file: both the camera_matrix and the distortion_ceofficients
/*
Function: saveCalibrationParameters
Description: This function saves the intrinsic camera parameters (camera matrix and distortion coefficients) to a file specified by the filename parameter. It uses OpenCV's FileStorage class to write the parameters to the file in XML or YAML format.

Parameters:
- cameraMatrix: Intrinsic camera matrix containing focal lengths and principal point.
- distCoeffs: Distortion coefficients (radial and tangential) of the camera.
- filename: Name of the file to which the parameters will be saved.

Returns: None
*/
void saveCalibrationParameters(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open " << filename << " for writing." << std::endl;
        return;
    }

    fs << "camera_matrix" << cameraMatrix << "distortion_coefficients" << distCoeffs;
    fs.release(); // Close the file

    std::cout << "Calibration parameters saved to " << filename << std::endl;
}


