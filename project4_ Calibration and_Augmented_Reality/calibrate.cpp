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


void DetectAndExtractTargetCorners(cv::Mat& frame, std::vector<cv::Point2f>& corners, const cv::Size& patternSize) {
    cv::Mat grayImage;
    cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);

    bool patternFound = findChessboardCorners(grayImage, patternSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

    if (patternFound) {
        cornerSubPix(grayImage, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
        drawChessboardCorners(frame, patternSize, cv::Mat(corners), patternFound);
    }
}


//Task 2
void saveCalibrationData(const std::vector<cv::Point2f>& corners, const cv::Size& patternSize) {
    corner_list.push_back(corners);

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



