/*
Zhimin Liang
Spring 2024
CS5330 Project 4

The calibrate.h file has the prototypes for all of the functions in calibrate.cpp.
*/

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <opencv2/ml.hpp>
#include <opencv2/ml/ml.hpp>
using namespace cv;
using namespace std;

#ifndef CALIBRETE_H
#define CALIBRETE_H 

// Define global variables to store corner and point information
extern std::vector<std::vector<cv::Point2f>> corner_list;
extern std::vector<std::vector<cv::Vec3f>> point_list;


void DetectAndExtractTargetCorners(cv::Mat& frame, std::vector<cv::Point2f>& corners,const Size& patternSize);
void saveCalibrationData(const vector<Point2f>& corners, const Size& patternSize) ;
void saveCalibrationParameters(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const std::string& filename);

#endif 