/*
Zhimin Liang
Spring 2024
CS5330 Project 3

The recognition.h file has the prototypes for all of the functions in recognition.cpp.
*/

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <opencv2/ml.hpp>
#include <opencv2/ml/ml.hpp>


#ifndef RECOGNITION_H
#define RECOGNITION_H 

struct RegionFeatures {
    cv::Moments moments;
    std::vector<double> huMoments;
    cv::Point centroid;
    double percentFilled;
    double aspectRatio;
    double orientation; 
};
struct ObjectFeatures {
    std::string label;
    std::vector<double> featureVector;
};


void thresholdToBinary(cv::Mat &src, cv::Mat &dst);
void cleanUpBinary(cv::Mat &src, cv::Mat &dst);
cv::Mat segmentation(const cv::Mat &cleanImage, cv::Mat &regionLabelMap, cv::Mat &stats, cv::Mat &centroids);
std::vector<RegionFeatures> computeRegionFeatures(cv::Mat &regionColorMap, cv::Mat &regionLabelMap, cv::Mat &stats, int minRegionSize);
void collectTrainingData(std::vector<double> featureVector, const std::string &label, const std::string &filename);
double scaledEuclideanDistance(const std::vector<double>& vec1,const std::vector<double>& vec2) ;
std::map<std::string, ObjectFeatures> loadKnownObjects(const std::string& filename);
std::string classifyNewFeatureVector(const std::vector<double>& newFeatureVector, const std::map<std::string, ObjectFeatures>& knownObjects);
void displayLabelOnStream(cv::Mat& frame, const std::string& label) ;
std::string classifyNewFeatureVectorKNN(const std::vector<double>& newFeatureVector, cv::Ptr<cv::ml::KNearest>& knn) ;

#endif 