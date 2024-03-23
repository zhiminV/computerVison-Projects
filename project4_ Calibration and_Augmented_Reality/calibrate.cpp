/*
Zhimin Liang
Spring 2024
CS5330 Project 3

Purpose: 
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


void DetectAndExtractTargetCorners(cv::Mat& frame, std::vector<cv::Point2f>& corners,const Size& patternSize) {
    Mat grayImage;
    cvtColor(frame, grayImage, COLOR_BGR2GRAY);

    bool patternFound = findChessboardCorners(grayImage, patternSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

    if (patternFound) {
        cornerSubPix(grayImage, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
        drawChessboardCorners(frame, patternSize, Mat(corners), patternFound);
    }
}


//Task 2
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


