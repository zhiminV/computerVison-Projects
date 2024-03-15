/*
Zhimin Liang
Spring 2024
CS5330 Project 3

Purpose: This file contains the implementation of an image processing and object recognition system. 
The code consists of functions for thresholding, cleaning up binary images, region segmentation, computing region features, collecting training data, calculating scaled Euclidean distance, 
loading known objects from a CSV file, classifying new feature vectors, displaying labels on output video streams, and classifying new feature vectors using K-Nearest Neighbors (KNN).
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "calibrate.h"

using namespace cv;
using namespace std;

std::vector<std::vector<cv::Point2f>> corner_list;
std::vector<std::vector<cv::Vec3f>> point_list;


void saveCalibrationData(const vector<Point2f>& corners, const Size& patternSize) {
    // Save the corner locations
    corner_list.push_back(corners);

    // Create and save the 3D world point set
    vector<Vec3f> point_set;
    for (int i = 0; i < corners.size(); ++i) {
        // Determine the z-coordinate based on the row index of the corner
        int row_index = i / patternSize.width;  // patternSize.width is the number of columns in the chessboard
        point_set.push_back(Vec3f(corners[i].x, corners[i].y, row_index));
    }
    point_list.push_back(point_set);
}


void DetectAndExtractTargetCorners(cv::Mat& frame, std::vector<cv::Point2f>& corners,const Size& patternSize) {
    Mat grayImage;
    cvtColor(frame, grayImage, COLOR_BGR2GRAY);

    bool patternFound = findChessboardCorners(grayImage, patternSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

    if (patternFound) {
        cornerSubPix(grayImage, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
        drawChessboardCorners(frame, patternSize, Mat(corners), patternFound);
    }
}