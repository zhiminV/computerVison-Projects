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

void DetectAndExtractTargetCorners(Mat& frame) {
    Size patternSize(9, 6); // checkerboard has 9 columns and 6 rows of internal corners
    Mat grayImage;
    cvtColor(frame, grayImage, COLOR_BGR2GRAY);

    vector<Point2f> corners;
    bool patternFound = findChessboardCorners(grayImage, patternSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

    if (patternFound) {
        cornerSubPix(grayImage, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
        drawChessboardCorners(frame, patternSize, Mat(corners), patternFound);
    }
}