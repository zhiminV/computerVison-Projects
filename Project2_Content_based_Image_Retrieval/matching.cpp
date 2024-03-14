/*
Zhimin Liang
Spring 2024
CS5330 Project 2

Purpose: This file implements various matching function  for image processing using OpenCV. 
*/
#include <opencv2/opencv.hpp>
#include <cstdio>
#include "matching.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

/*
Purpose: Compute 7x7 square features for a given image.
   Arguments:
      - const cv::Mat& image - Input image.
   Returns:
      - std::vector<float> - Feature vector.
*/

std::vector<float> computeFeatures(const cv::Mat& image) {
    int rows = image.rows;
    int cols = image.cols;

    if (rows < 7 || cols < 7) {
        std::cerr << "Error: Input image is too small!" << std::endl;
        return std::vector<float>();
    }

    int centerX = cols / 2;
    int centerY = rows / 2;

    int startX = std::max(0, centerX - 3);
    int startY = std::max(0, centerY - 3);

    int endX = std::min(cols, centerX + 4);
    int endY = std::min(rows, centerY + 4);

    // Create a vector to store the feature vector (7x7 square) with color information
    std::vector<float> featureVector;

    // Extract the feature vector (7x7 square) from the input image
    for (int y = startY; y < endY; ++y) {
        for (int x = startX; x < endX; ++x) {
            // Extract color channels (assuming image is in BGR format)
            for (int c = 0; c < image.channels(); ++c) {
                featureVector.push_back(image.at<cv::Vec3b>(y, x)[c]);
            }
        }
    }

    return featureVector;
}

/*
Purpose: Compute sum-of-squared-difference distance between two feature vectors.
   Arguments:
      - const std::vector<float>& features1 - First feature vector.
      - const std::vector<float>& features2 - Second feature vector.
   Returns:
      - float - Sum-of-squared-difference distance.
*/
float computeSSDDistance(const std::vector<float>& features1, const std::vector<float>& features2) {
    // Ensure that the feature vectors are of the same size
    if (features1.size() != features2.size()) {
        std::cerr << "Feature vectors have different sizes!" << std::endl;
        return -1.0f; 
    }

    // Compute sum-of-squared-difference distance
    float distance = 0.0f;
    for (size_t i = 0; i < features1.size(); ++i) {
        distance += std::pow(features1[i] - features2[i], 2);
    }

    return distance;
}

/* Purpose: Compute histogram from an image.
   Arguments:
      - const cv::Mat& image - Input image.
      - int bins - Number of bins for the histogram.
   Returns:
      - std::vector<float> - Computed histogram.
*/
std::vector<float> computeHistogram(const cv::Mat& image, int bins) {
    int histsize = bins;
    float max;

    // Initialize the histogram (use floats so we can make probabilities)
    std::vector<float> hist(histsize * histsize, 0.0f);

    // Keep track of largest bucket for visualization purposes
    max = 0;

    // Loop over all pixels
    for (int i = 0; i < image.rows; i++) {
        const cv::Vec3b* ptr = image.ptr<cv::Vec3b>(i); // Pointer to row i
        for (int j = 0; j < image.cols; j++) {

            // Get the RGB values
            float B = ptr[j][0];
            float G = ptr[j][1];
            float R = ptr[j][2];

            // Compute the r, g chromaticity
            float divisor = R + G + B;
            divisor = divisor > 0.0 ? divisor : 1.0; // Check for all zeros
            float r = R / divisor;
            float g = G / divisor;

            // Compute indexes, r, g are in [0, 1]
            int rindex = static_cast<int>(r * (histsize - 1) + 0.5);
            int gindex = static_cast<int>(g * (histsize - 1) + 0.5);

            // Increment the histogram
            hist[rindex * histsize + gindex]++;

            // Keep track of the size of the largest bucket (just so we know what it is)
            float newvalue = hist[rindex * histsize + gindex];
            max = newvalue > max ? newvalue : max;
        }
    }

    // Normalize the histogram by the number of pixels
    for (float& value : hist) {
        value /= (image.rows * image.cols);
    }

    return hist;
}


/*
Purpose: Compute features for the top and bottom halves of the image.
   Arguments:
      - const cv::Mat& image - Input image.
      - int bins - Number of bins for the histograms.
   Returns:
      - std::vector<float> - Combined histogram features.
*/
std::vector<float> computeTwoHistograms(const cv::Mat& image, int bins) {
    int rows = image.rows;
    int cols = image.cols;

    // Check if the image size is sufficient
    if (rows < 2 || cols < 1) {
        std::cerr << "Error: Input image is too small!" << std::endl;
        return std::vector<float>();
    }

    // Create histograms for the top and bottom halves
    std::vector<float> topHist(bins * bins, 0.0f);
    std::vector<float> bottomHist(bins * bins, 0.0f);

    // Compute the height of each half
    int halfHeight = rows / 2;

    
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            // Extract color channels 
            for (int c = 0; c < image.channels(); ++c) {
                float value = static_cast<float>(image.at<cv::Vec3b>(y, x)[c]);

                int index = c * bins + static_cast<int>(value * bins / 256.0);  // Adjusted bin calculation
                if (y < halfHeight) {
                    topHist[index]++;
                } else {
                    bottomHist[index]++;
                }
            }
        }
    }

    // Normalize histograms by the number of pixels
    float totalPixels = static_cast<float>(rows * cols);
    for (float& value : topHist) {
        value /= totalPixels;
    }
    for (float& value : bottomHist) {
        value /= totalPixels;
    }

    // Combine the two histograms
    std::vector<float> combinedHist;
    combinedHist.insert(combinedHist.end(), topHist.begin(), topHist.end());
    combinedHist.insert(combinedHist.end(), bottomHist.begin(), bottomHist.end());

    return combinedHist;
}


/*
Purpose: Compute histogram intersection as the distance metric.
   Arguments:
      - const std::vector<float>& hist1 - First histogram.
      - const std::vector<float>& hist2 - Second histogram.
   Returns:
      - float - Histogram intersection distance.
*/
float computeHistogramIntersection(const std::vector<float>& hist1, const std::vector<float>& hist2){
    if (hist1.size() != hist2.size()) {
        std::cerr << "Histograms have different sizes" << std::endl;
        return -1.0f; 
    }

    float intersection = 0.0f;
    for (int i = 0; i < hist1.size(); ++i) {
        intersection += std::min(hist1[i], hist2[i]);
       
    }
   
    return 1.0f - intersection; //0.0 represents maximum similarity and 1.0 represents no similarity
}

/*Applies separable 3x3 Sobel X and 3x3 Sobel Y filters to the input image.
Arguments:
 cv::Mat &src - Input image.
  cv::Mat &dst - Output image with applied Sobel filter.
  
Returns:
  0 on successful execution.
*/

/*-1  0  1
-2  0  2
-1  0  1*/
int sobelX3x3( const cv::Mat &src, cv::Mat &dst)  {
    dst.create(src.size(), CV_16SC3);
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            for (int c = 0; c < src.channels(); c++) {
                int sobelX = -src.at<cv::Vec3b>(y - 1, x - 1)[c] +
                             src.at<cv::Vec3b>(y - 1, x + 1)[c] -
                             2 * src.at<cv::Vec3b>(y, x - 1)[c] +
                             2 * src.at<cv::Vec3b>(y, x + 1)[c] -
                             src.at<cv::Vec3b>(y + 1, x - 1)[c] +
                             src.at<cv::Vec3b>(y + 1, x + 1)[c];

                dst.at<cv::Vec3s>(y, x)[c] = static_cast<short>(sobelX);
            }
        }
    }

    return 0;
}

/*
Implement a 3x3 Sobel Y filter as separable 1x3 filters
Applies a separable 3x3 Sobel Y filter to the input image.
Arguments:
  cv::Mat &src - Input image.
  cv::Mat &dst - Output image with applied Sobel filter.
Returns:
  0 on successful execution.
*/

/*-1 -2 -1
 0  0  0
 1  2  1*/
int sobelY3x3( const cv::Mat &src, cv::Mat &dst)  {
    dst.create(src.size(), CV_16SC3);

    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            for (int c = 0; c < src.channels(); c++) {
                int sobelY = -src.at<cv::Vec3b>(y - 1, x - 1)[c] -
                             2 * src.at<cv::Vec3b>(y - 1, x)[c] -
                             src.at<cv::Vec3b>(y - 1, x + 1)[c] +
                             src.at<cv::Vec3b>(y + 1, x - 1)[c] +
                             2 * src.at<cv::Vec3b>(y + 1, x)[c] +
                             src.at<cv::Vec3b>(y + 1, x + 1)[c];

                dst.at<cv::Vec3s>(y, x)[c] = static_cast<short>(sobelY);
            }
        }
    }

    return 0;
}


/*
 Implement a function that generates a gradient magnitude image from the X and Y Sobel images
Generates a gradient magnitude image from the input Sobel X and Y images.
Arguments:
  cv::Mat &sx - Input Sobel X image.
  cv::Mat &sy - Input Sobel Y image.
  cv::Mat &dst - Output gradient magnitude image.

Returns:
  0 on successful execution.
*/
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst){
    CV_Assert(sx.size() == sy.size() && sx.type() == CV_16SC3 && sy.type() == CV_16SC3);

    dst.create(sx.size(), CV_8UC3);

    for (int y = 0; y < sx.rows; y++) {
        for (int x = 0; x < sx.cols; x++) {
            for (int c = 0; c < sx.channels(); c++) {
                // Calculate gradient magnitude using Euclidean distance
                double mag = std::sqrt(static_cast<double>(sx.at<cv::Vec3s>(y, x)[c]) * sx.at<cv::Vec3s>(y, x)[c] +
                                       static_cast<double>(sy.at<cv::Vec3s>(y, x)[c]) * sy.at<cv::Vec3s>(y, x)[c]);

                dst.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(mag);
            }
        }
    }

    return 0;
}


/* Purpose: Calculate the Sobel magnitude image and use a histogram of gradient magnitudes as a texture feature.
   Arguments:
      - const cv::Mat& image - Input image.
      - int bins - Number of bins for the histogram.
   Returns:
      - std::vector<float> - Texture histogram feature.
*/
std::vector<float> computeTextureHistogram(const cv::Mat &image, int bins) {
    cv::Mat sobelX, sobelY;
    sobelX3x3(image, sobelX);
    sobelY3x3(image, sobelY);

    cv::Mat gradientMagnitude;
    magnitude(sobelX, sobelY, gradientMagnitude);
    cv::normalize(gradientMagnitude, gradientMagnitude, 0, 255, cv::NORM_MINMAX);

    std::vector<float> textureHist(bins, 0.0f);

    for (int i = 0; i < gradientMagnitude.rows; ++i) {
        const uchar *ptr = gradientMagnitude.ptr<uchar>(i);
        for (int j = 0; j < gradientMagnitude.cols; ++j) {
            int index = static_cast<int>(ptr[j] * bins / 256.0);
            textureHist[index]++;
        }
    }

    float totalPixels = static_cast<float>(gradientMagnitude.rows * gradientMagnitude.cols);
    for (float &value : textureHist) {
        value /= totalPixels;
    }

    return textureHist;
}

/*
 Purpose: Load DNN embeddings from a CSV file.
    Arguments:
       - const std::string& csvFilePath - Path to the CSV file.
       - const std::string& targetImage - Target image for which embeddings are to be loaded.
    Returns:
       - std::vector<float> - Loaded DNN embeddings.
*/
std::vector<float> readDNNEmbeddings(const std::string& csvFilePath, const std::string& targetImage) {
    std::vector<float> targetEmbeddings;

    std::ifstream file(csvFilePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open CSV file: " << csvFilePath << std::endl;
        return targetEmbeddings;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string imageName;
        std::getline(iss, imageName, ',');

        if (imageName == targetImage) {
            targetEmbeddings.clear(); // Clear previous embeddings if any

            float value;
            while (iss >> value) {
                targetEmbeddings.push_back(value);
                if (iss.peek() == ',') {
                    iss.ignore();
                }
            }

            // Successfully loaded target embeddings
            break;
        }
    }

    file.close();

    return targetEmbeddings;
}


/*
 Purpose: Compute combined features using texture and DNN embeddings.
    Arguments:
       - const cv::Mat& image - Input image.
       - const std::string& dnnCsvFilePath - Path to the CSV file containing DNN embeddings.
       - const std::string& targetImage - Target image for which features are to be computed.
    Returns:
       - std::vector<float> - Combined feature vector.
*/
std::vector<float> computeCombinedFeatures(const cv::Mat& image, const std::string& dnnCsvFilePath, const std::string& targetImage) {
    // Compute texture features
    int bins = 16;
    std::vector<float> textureHist = computeTextureHistogram(image, bins);

    // Compute DNN embeddings
    std::vector<float> dnnEmbeddings = readDNNEmbeddings(dnnCsvFilePath, targetImage);

    // Combine texture histogram and DNN embeddings into a unified feature vector
    std::vector<float> combinedFeatures;
    combinedFeatures.insert(combinedFeatures.end(), textureHist.begin(), textureHist.end());
    combinedFeatures.insert(combinedFeatures.end(), dnnEmbeddings.begin(), dnnEmbeddings.end());

    return combinedFeatures;
}

/*
   Purpose: Compute cosine distance between two feature vectors.
    Arguments:
       - const std::vector<float>& features1 - First feature vector.
       - const std::vector<float>& features2 - Second feature vector.
    Returns:
       - float - Cosine distance.
*/
float computeCosineDistance(const std::vector<float>& features1, const std::vector<float>& features2) {
    // Ensure that the feature vectors are of the same size
    if (features1.size() != features2.size()) {
        std::cerr << "Feature vectors have different sizes!" << std::endl;
        return -1.0f; 
    }

    // Compute dot product
    float dotProduct = 0.0f;
    for (size_t i = 0; i < features1.size(); ++i) {
        dotProduct += features1[i] * features2[i];
    }

    // Compute magnitudes
    float magnitude1 = 0.0f, magnitude2 = 0.0f;
    for (size_t i = 0; i < features1.size(); ++i) {
        magnitude1 += std::pow(features1[i], 2);
        magnitude2 += std::pow(features2[i], 2);
    }

    // Avoid division by zero
    if (magnitude1 == 0 || magnitude2 == 0) {
        return 1.0f; // Return maximum distance if either magnitude is zero
    }

    // Compute cosine similarity
    float cosineSimilarity = dotProduct / (std::sqrt(magnitude1) * std::sqrt(magnitude2));

    // Compute cosine distance (1 - cosine similarity)
    return 1.0f - cosineSimilarity;
}


