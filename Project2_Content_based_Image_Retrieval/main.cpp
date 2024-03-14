
/*
Zhimin Liang
Spring 2024
CS5330 Project 2

Purpose: This file contains the main program 
*/

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <iostream>
#include "matching.h"
#include <vector>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <filesystem>

int main(int argc, char *argv[]) {
    char dirname[256];
    char buffer[256];
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;

    // check for sufficient arguments
    if (argc < 2) {
        printf("usage: %s <directory path>\n", argv[0]);
        exit(-1);
    }

    // get the directory path
    strcpy(dirname, argv[1]);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    // // task 1
    // cv::Mat targetImage = cv::imread("/Users/lzm/Desktop/5330 CV/project/Project2/olympus/pic.1016.jpg", cv::IMREAD_COLOR);
    // std::vector<float> targetFeatures = computeFeatures(targetImage);

    // //Task 2
    // cv::Mat targetImage = cv::imread("/Users/lzm/Desktop/5330 CV/project/Project2/olympus/pic.0164.jpg", cv::IMREAD_COLOR);
    // std::vector<float> targetHis = computeHistogram(targetImage,8);

    // //Task 3
    // cv::Mat targetImage = cv::imread("/Users/lzm/Desktop/5330 CV/project/Project2/olympus/pic.0274.jpg", cv::IMREAD_COLOR);
    // std::vector<float> targetHis = computeTwoHistograms(targetImage,16);

    // //Task 4
    // cv::Mat targetImage = cv::imread("/Users/lzm/Desktop/5330 CV/project/Project2/olympus/pic.0535.jpg", cv::IMREAD_COLOR);
    // std::vector<float> targetColorHist = computeHistogram(targetImage, 16);
    // std::vector<float> targetTextureHist = computeTextureHistogram(targetImage, 16);

    // //Task 5
    // std::string csvFilePath = argv[2];
    // std::string targetImage = "pic.0893.jpg"; //can change other target like pic.0164.jpg 
    // std::vector<float> targetEmbeddings = readDNNEmbeddings(csvFilePath, targetImage);

    // if (targetEmbeddings.empty()) {
    //     std::cerr << "Target image not found in the CSV file!" << std::endl;
    //     return -1;
    // }

    //Task 7
    std::string csvFilePath = argv[2];
    std::string targetImage = "pic.0375.jpg"; 
    cv::Mat target = cv::imread("/Users/lzm/Desktop/5330 CV/project/Project2/olympus/pic.0375.jpg", cv::IMREAD_COLOR);
    std::vector<float> targetCombine = computeCombinedFeatures(target, csvFilePath, targetImage);



    // Do not comment this line, use for evey task
    std::vector<std::pair<std::string, float>> matches; 

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {

            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            // Read the current image
            cv::Mat currentImage = cv::imread(buffer, cv::IMREAD_COLOR);

            // // Task 1 
            // std::vector<float> currentFeatures = computeFeatures(currentImage);
            // float distance = computeSSDDistance(targetFeatures, currentFeatures);

            // //Task 2
            // std::vector<float>  currentHis = computeHistogram(currentImage,8);
            // float distance = computeHistogramIntersection(targetHis, currentHis);

            // //Task 3
            // std::vector<float>  currentHis = computeTwoHistograms(currentImage,16);
            // float distance = computeHistogramIntersection(targetHis, currentHis);

            // //Task 4 
            // std::vector<float> currentColorHist = computeHistogram(currentImage, 16);
            // std::vector<float> currentTextureHist = computeTextureHistogram(currentImage, 16);
            // float colorDistance = computeSSDDistance(targetColorHist, currentColorHist);
            // float textureDistance = computeSSDDistance(targetTextureHist, currentTextureHist);
            // float distance = 0.5f * (colorDistance + textureDistance);

            // //Task 5
            // std::vector<float> currentEmbeddings = readDNNEmbeddings(csvFilePath, dp->d_name);
            // float distance = computeSSDDistance(targetEmbeddings, currentEmbeddings);

            //Task 7
            std::vector<float> currentCombine = computeCombinedFeatures(currentImage,csvFilePath, dp->d_name);
            float distance = computeCosineDistance(targetCombine, currentCombine);

            // Do not comment this line, use for evey task
            matches.push_back({buffer, distance});

            
        }
    }

    // Sort the list of matches based on distance (ascending order)
    std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    int topN = 6;
    std::cout << "Top " << topN << " matches:" << std::endl;
    for (int i = 0; i < topN; ++i) {
        std::cout << "Image: " << matches[i].first << ", Distance: " << matches[i].second << std::endl;
      
        cv::Mat topImage = cv::imread(matches[i].first, cv::IMREAD_COLOR);
        if (!topImage.empty()) {
            cv::imshow("Top Match", topImage);
            cv::waitKey(0); 
        } else {
            std::cerr << "Error: Unable to load image " << matches[i].first << std::endl;
        }
    }
    return 0;
}

