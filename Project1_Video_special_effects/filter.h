/*
Zhimin Liang
Spring 2024
CS5330 Project 1

The filters.h file has the prototypes for all of the functions in filter.cpp.
*/

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#ifndef VID_DISPLAY_H
#define VID_DISPLAY_H
#include "faceDetect.h"

int greyscale(cv::Mat &src, cv::Mat &dst);
int sepiaTone(cv::Mat &src, cv::Mat &dst);
int sepiawithVignette(cv::Mat &src, cv::Mat &dst);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);
int blurOutsideFaces(cv::Mat &src, cv::Mat &dst);
int embossingEffect(cv::Mat &src, cv::Mat &dst);
void addSparklesAboveFaces(cv::Mat &frame, const std::vector<cv::Rect> &faces);
void cartoonize(cv::Mat &input, cv::Mat &output);

#endif