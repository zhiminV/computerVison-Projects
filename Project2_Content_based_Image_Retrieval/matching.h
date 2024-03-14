/*
Zhimin Liang
Spring 2024
CS5330 Project 2

The matching.h file has the prototypes for all of the functions in matching.cpp.
*/
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>
#include <algorithm>
#ifndef MATCHING_H
#define MATCHING_H

std::vector<float> computeFeatures(const cv::Mat& image);

float computeSSDDistance(const std::vector<float>& features1, const std::vector<float>& features2);

std::vector<float> computeHistogram(const cv::Mat& image, int bins);

std::vector<float> computeTwoHistograms(const cv::Mat& image, int bins);

float computeHistogramIntersection(const std::vector<float>& hist1, const std::vector<float>& hist2);

int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

std::vector<float> computeTextureHistogram(const cv::Mat& image, int bins);
std::vector<float> readDNNEmbeddings(const std::string& csvFilePath, const std::string& targetImage);
cv::Mat computeGaborFeatures(const cv::Mat& image);
std::vector<float> computeCombinedFeatures(const cv::Mat& image, const std::string& dnnCsvFilePath, const std::string& targetImage);
float computeCosineDistance(const std::vector<float>& features1, const std::vector<float>& features2);


#endif 