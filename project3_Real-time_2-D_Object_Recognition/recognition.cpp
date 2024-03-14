/*
Zhimin Liang
Spring 2024
CS5330 Project 3

 Purpose: This file contains the implementation of an image processing and object recognition system. 
 The code consists of functions for thresholding, cleaning up binary images, region segmentation, computing region features, collecting training data, calculating scaled Euclidean distance, 
 loading known objects from a CSV file, classifying new feature vectors, displaying labels on output video streams, and classifying new feature vectors using K-Nearest Neighbors (KNN).


*/

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include "recognition.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/ml.hpp>
#include <opencv2/ml/ml.hpp>



/*
Purpose: Threshold the input image into a binary image.

Arguments:
    - cv::Mat& src - Input image.
    - cv::Mat& dst - Output binary image.


*/
void thresholdToBinary(cv::Mat &src, cv::Mat &dst){
    if (src.channels() != 3) {
        return;
    }

    uchar threshold = 120;
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);// make the image to gray, which
    //8-bint interger values in the range 0 to 255,a single channel
    dst = cv::Mat(src.size(), CV_8UC1); 
   

    for(int i = 0; i < gray.rows; i++){
        for(int j = 0; j < gray.cols; j++){
            uchar intensity = gray.at<uchar>(i, j);
            //, intensity typically refers to the brightness level of a pixel
            //at pixels with lower brightness (darker) are considered part of an object.
            if(intensity <= threshold){
                dst.at<uchar>(i,j) = 255; 
            }
            else{
                dst.at<uchar>(i,j) = 0; // otherwide, it it the background, and make it black
            }
        }
    }
    
}

    
/*
Purpose: Clean up the binary image using a median filter and erosion.

Arguments:
    - cv::Mat& src - Input binary image.
    - cv::Mat& dst - Output cleaned binary image.

*/ 
void cleanUpBinary(cv::Mat &src, cv::Mat &dst){

    dst = src.clone();

    int threshold = 255*5;// the pixel just 0 or 255, so if sum > this, meansthe median is 255
    // assume use 3 by 3 box
    for(int x = 1 ; x < src.rows -1; x++){
        for(int y = 1; y < src.cols - 1; y++){
            int sum = 0;
            // sum pixel values in the 3x3 neighborhood
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    sum += src.at<uchar>(i + x, j + y);
                } 
            
            }
            // replace the center pixel with median 
            dst.at<uchar>(x, y) = (sum >= threshold) ? 255 : 0;
        }
        
    }
 // Use erosion to shrink white regions
    cv::Mat erosionKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(dst, dst, erosionKernel);

}

/*
Purpose: Perform region segmentation on a cleaned binary image.

Arguments:
    - const cv::Mat& cleanImage - Input cleaned binary image.
    - cv::Mat& regionLabelMap - Output matrix containing labeled regions.
    - cv::Mat& stats - Output statistics matrix for each labeled region.
    - cv::Mat& centroids - Output matrix containing the centroids of labeled regions.

Returns:
    - cv::Mat - Colored region map.

*/
cv::Mat segmentation(const cv::Mat &cleanImage, cv::Mat &regionLabelMap, cv::Mat &stats, cv::Mat &centroids) {
    int maxNumRegions = 3; // Limit recognition to the largest N regions (e.g., top 3)
    int minRegionSize = 4500;

    // Perform connected components labeling
    int numRegions = cv::connectedComponentsWithStats(cleanImage, regionLabelMap, stats, centroids);

    // Vector to store RGB color for each region
    std::vector<cv::Vec3b> colorMap(numRegions, cv::Vec3b(0, 0, 0));

    for (int label = 1; label < numRegions; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);

        if (area >= minRegionSize) {
            // Assign a consistent color based on label
            colorMap[label] = cv::Vec3b((label * 131) % 256, (label * 71) % 256, (label * 211) % 256);
        }
    }

    // Create a colored region map
    cv::Mat regionColorMap = cv::Mat::zeros(regionLabelMap.size(), CV_8UC3);
    for (int row = 0; row < regionColorMap.rows; ++row) {
        for (int col = 0; col < regionColorMap.cols; ++col) {
            int label = regionLabelMap.at<int>(row, col);
            regionColorMap.at<cv::Vec3b>(row, col) = colorMap[label];
        }
    }

    return regionColorMap;
}


/*
Purpose: Compute region features, draw bounding boxes, and orientation arrows.

Arguments:
    - cv::Mat& regionColorMap - Colored region map.
    - cv::Mat& regionLabelMap - Matrix containing labeled regions.
    - cv::Mat& stats - Statistics matrix for each labeled region.
    - int minRegionSize - Minimum region size for recognition.

Returns:
    - std::vector<RegionFeatures> - Vector containing region features.

*/
std::vector<RegionFeatures> computeRegionFeatures(cv::Mat &regionColorMap, cv::Mat &regionLabelMap, cv::Mat &stats, int minRegionSize) {
    std::vector<RegionFeatures> features; // Vector to store region features
    for (int label = 1; label < regionLabelMap.rows; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);

        if (area >= minRegionSize) {
            // Convert regionLabelMap to a binary image for the specific label
            cv::Mat regionBinary = (regionLabelMap == label);

            // Extract contour for the region
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(regionBinary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            // Fit rotated bounding box to the contour
            if (!contours.empty()) {
                cv::RotatedRect rotatedRect = cv::minAreaRect(contours[0]);

                // Draw rotated bounding box on the regionColorMap with color
                cv::Point2f vertices[4];
                rotatedRect.points(vertices);
                for (int i = 0; i < 4; ++i) {
                    cv::line(regionColorMap, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 255, 255), 2);
                }

                // Compute the moments for the region
                cv::Moments moments = cv::moments(contours[0]);
                 // Calculate the centroid
                cv::Point centroid(static_cast<int>(moments.m10 / moments.m00),
                                    static_cast<int>(moments.m01 / moments.m00));

                // Calculate the orientation angle (theta) using the atan2 function
                double theta = 0.5 * atan2(2.0 * moments.mu11, moments.mu20 - moments.mu02);
                // Calculate the axis of least central moment
                cv::Point vectorEnd(static_cast<int>(centroid.x + 50 * cos(theta)),
                                    static_cast<int>(centroid.y + 50 * sin(theta)));
                double scaleFactor = 1.5; // Adjust the scale factor as needed
                cv::Point2f extendedVectorEnd = centroid + scaleFactor * (vectorEnd - centroid);


                cv::arrowedLine(regionColorMap, centroid,extendedVectorEnd, cv::Scalar(0, 255, 0), 5);

                // Extract bounding box information from the stats matrix
                int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
                int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);

                // Calculate the percent filled
                double percentFilled = moments.m00 / (width * height) * 100.0;

                // Calculate bounding box height/width ratio
                double aspectRatio = static_cast<double>(height) / width;

                // Calculate Hu moments
                std::vector<double> huMoments;
                cv::HuMoments(moments, huMoments);

                // Store the features in a vector of structures
                RegionFeatures regionFeature;
                regionFeature.moments = moments;
                regionFeature.huMoments = huMoments;
                regionFeature.centroid = centroid;
                regionFeature.orientation = theta;
                regionFeature.percentFilled = percentFilled;
                regionFeature.aspectRatio = aspectRatio;
                features.push_back(regionFeature);

                // Draw text on the regionColorMap with feature information
                std::stringstream featureText;
                featureText << "Filled: " << percentFilled << "%"<< "\n";
                featureText <<  "AspectRatio: " << aspectRatio << "\n";

                cv::putText(regionColorMap, featureText.str(), cv::Point(100,200), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
            }
        }
    }

    return features;
}

/*
Purpose: Collect training data and append it to a file.

Arguments:
    - std::vector<double> featureVector - Feature vector.
    - const std::string& label - Label associated with the feature vector.
    - const std::string& filename - File to which training data is appended.

*/
void collectTrainingData(std::vector<double> featureVector, const std::string &label, const std::string &filename) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error opening file for training data." << std::endl;
        return;
    }
 
    file << label;
    // Write feature vector and label to the file
    for (const auto &feature : featureVector) {
        file << "," << feature;
    }
    file << std::endl;

    file.close();
}


/*
Purpose: Calculate the scaled Euclidean distance between two feature vectors.

Arguments:
    - const std::vector<double>& vec1 - First feature vector.
    - const std::vector<double>& vec2 - Second feature vector.

Returns:
    - double - Scaled Euclidean distance.

*/
double scaledEuclideanDistance(const std::vector<double>& vec1, const std::vector<double>& vec2) {

    double distance = 0.0;
    // Calculate the sum of squared differences for each dimension
    for (size_t i = 0; i < vec1.size(); ++i) {
        distance += std::pow((vec1[i] - vec2[i]), 2.0);
    }

    // Calculate the scaled Euclidean distance
    return std::sqrt(distance) / static_cast<double>(vec1.size());
}



/*
Purpose: Load known objects and their features from a CSV file.

Arguments:
    - const std::string& filename - CSV file containing known objects and features.

Returns:
    - std::map<std::string, ObjectFeatures> - Map of known objects and their features.

*/
std::map<std::string, ObjectFeatures> loadKnownObjects(const std::string& filename) {
    std::map<std::string, ObjectFeatures> knownObjects;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return knownObjects;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        ObjectFeatures obj;

        // Read label
        if (!std::getline(ss, obj.label, ',')) {
            // Skip lines without a label
            continue;
        }

        // Read features
        double feature;
        while (ss >> feature) {
            obj.featureVector.push_back(feature);

            // Check for comma, and ignore it
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }

        // Check if features are empty, and skip this entry
        if (obj.featureVector.empty()) {
            continue;
        }
       
        knownObjects[obj.label] = obj;
    }

    file.close();
    return knownObjects;
}


/*
Purpose: Classify a new feature vector based on scaled Euclidean distance.

Arguments:
    - const std::vector<double>& newFeatureVector - Feature vector to be classified.
    - const std::map<std::string, ObjectFeatures>& knownObjects - Map of known objects and their features.

Returns:
    - std::string - Label of the closest known object.

*/
std::string classifyNewFeatureVector(const std::vector<double>& newFeatureVector, const std::map<std::string, ObjectFeatures>& knownObjects) {
    double minDistance = std::numeric_limits<double>::max();
    std::string closestLabel;

    for (const auto& knownObject : knownObjects) {

        double distance = scaledEuclideanDistance(newFeatureVector, knownObject.second.featureVector);

        if (distance < minDistance) {
            minDistance = distance;
            closestLabel = knownObject.first;
            
        }
    }

    return closestLabel;
}


/*
Purpose: Display the classified label on the output video stream.

Arguments:
    - cv::Mat& frame - Input video frame.
    - const std::string& label - Classified label.

*/
void displayLabelOnStream(cv::Mat& frame, const std::string& label) {
    cv::putText(frame, "Label: " + label, cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
}


/*
Purpose: Classify a new feature vector using the K-Nearest Neighbors (KNN) algorithm.

Arguments:
    - const std::vector<double>& newFeatureVector - Feature vector to be classified.
    - cv::Ptr<cv::ml::KNearest>& knn - KNN model trained on known objects.

Returns:
    - std::string - Label of the closest known object.

*/
std::string classifyNewFeatureVectorKNN(const std::vector<double>& newFeatureVector, cv::Ptr<cv::ml::KNearest>& knn) {
    // Convert the feature vector to a cv::Mat
    cv::Mat testSample(1, static_cast<int>(newFeatureVector.size()), CV_32F);
    for (size_t i = 0; i < newFeatureVector.size(); ++i) {
        testSample.at<float>(0, static_cast<int>(i)) = static_cast<float>(newFeatureVector[i]);
    }
    // std::cout << "Test Samples: " << testSample.rows << "x" << testSample.cols << ", Type: " << testSample.type() << std::endl;
    // Ensure the input feature vector has the same dimension as the training data
    if (testSample.cols != knn->getVarCount()) {
        std::cerr << "Error: Feature vector size mismatch with the training data." << std::endl;
        return "";
    }
    // Use KNN to find the k nearest neighbors
    cv::Mat results, neighborResponses, dists;
    knn->findNearest(testSample, 3, results, neighborResponses, dists);

    // Count the occurrences of each label in the neighbors
    std::map<std::string, int> labelCount;
    for (int i = 0; i < results.rows; ++i) {
        std::string label = std::to_string(static_cast<int>(neighborResponses.at<float>(i, 0)));
        labelCount[label]++;
    }

    // Find the label with the highest count
    int maxCount = 0;
    std::string maxLabel;
    for (const auto& entry : labelCount) {
        if (entry.second > maxCount) {
            maxCount = entry.second;
            maxLabel = entry.first;
        }
    }

    return maxLabel;
}
