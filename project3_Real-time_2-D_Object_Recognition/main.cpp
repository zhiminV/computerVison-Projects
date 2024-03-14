/*
Zhimin Liang
Spring 2024
CS5330 Project 3

Purpose: This file contains the main program 
*/

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include "recognition.h"
#include <string>
#include <opencv2/ml.hpp>
#include <opencv2/ml/ml.hpp>


int main(int argc, char *argv[]) {
        cv::VideoCapture *capdev;

        // open the video device
        capdev = new cv::VideoCapture(0);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }

        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Binary Image", 1); // identifies a window
        cv::Mat frame, blur_out, binaryImage, cleanImage, regionColorMap, regionLabelMap, stats, centroids;
        int minRegionSize = 6000;
        std::string filename = "objectDB.csv";
        std::map<std::string, ObjectFeatures> knownObjects = loadKnownObjects(filename);

        // Task 9: Implement a second classification method - K-Nearest Neighbors with K = 3
       // Convert known objects to training samples
        cv::Mat trainingSamples(static_cast<int>(knownObjects.size()), static_cast<int>(knownObjects.begin()->second.featureVector.size()), CV_32F);
        cv::Mat labels(static_cast<int>(knownObjects.size()), 1, CV_32F);  // Use CV_32F for floating-point labels

        // Update the conversion of label in the main function
      
        // Populate trainingSamples
        int row = 0;
        for (const auto& knownObject : knownObjects) {
                std::string label = knownObject.first;

                for (size_t i = 0; i < knownObject.second.featureVector.size(); ++i) {
                        trainingSamples.at<float>(row, static_cast<int>(i)) = static_cast<float>(knownObject.second.featureVector[i]);
                }

                row++;
        }

        // Create KNN model
        cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();

        // Set K = 3
        knn->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
        knn->setDefaultK(3);

        // Train the KNN model
        cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(trainingSamples, cv::ml::ROW_SAMPLE, labels);
        knn->train(td);



        while(true) {
                 *capdev >> frame; // get a new frame from camera
                if( frame.empty() ) {
                  printf("frame is empty\n");
                  break;
                } 
                
                //blurring the image a little can make the regions more uniform.
                cv::GaussianBlur(frame, blur_out, cv::Size(1, 1), 2.0, 2.0);
                
                thresholdToBinary(blur_out, binaryImage);
                cleanUpBinary(binaryImage, cleanImage);
                // cv::imshow("original image", frame);
                // cv::imshow("blur image", blur_out);
                // cv::imshow("binary image", binaryImage);
                // cv::imshow("clean image", cleanImage);

                char key = cv::waitKey(10);
                if( key == 'q') {
                    break;
                }
                //Task 4
                else if(key == 'r'){
                    regionColorMap = segmentation(cleanImage, regionLabelMap, stats,centroids);
                    std::vector<RegionFeatures> regionFeatures = computeRegionFeatures(regionColorMap,regionLabelMap,stats,minRegionSize);
                    cv::imshow("Region Boxes", regionColorMap);
          
                }
                //Task 5
                else if(key == 'n'){
                        std::string label;
                        std::cout << "Please label the object: ";
                        std::cin >> label; //read from standard input
                        regionColorMap = segmentation(cleanImage, regionLabelMap, stats,centroids);
                        std::vector<RegionFeatures> regionFeatures = computeRegionFeatures(regionColorMap,regionLabelMap,stats,minRegionSize);
                        cv::imshow("Region Boxes", regionColorMap);
                        std::vector<double> featureVector;
                        for (const auto &feature : regionFeatures) {
                                std::vector<double> huMoments;
                                cv::HuMoments(feature.moments, huMoments);
                                for (const auto &hu : huMoments) {
                                        featureVector.push_back(hu); // Add each Hu1, Hu2, ..., Hu7 Hu Moment to the feature vector
                                }
                               // For cv::Point (centroid), add its x and y coordinates
                                featureVector.push_back(static_cast<double>(feature.centroid.x));
                                featureVector.push_back(static_cast<double>(feature.centroid.y));
                                featureVector.push_back(feature.percentFilled);
                                featureVector.push_back(feature.aspectRatio);
                                featureVector.push_back(feature.orientation);
                        }
                        collectTrainingData(featureVector,label,filename);

                }
                //Task 6
                // Get the new feature vector for classification
                regionColorMap = segmentation(cleanImage, regionLabelMap, stats,centroids);
                std::vector<RegionFeatures> regionFeatures = computeRegionFeatures(frame,regionLabelMap,stats,minRegionSize);
                cv::imshow("Region Boxes", regionColorMap);
                std::vector<double> featureVector;
                for (const auto &feature : regionFeatures) {
                        std::vector<double> huMoments;
                        cv::HuMoments(feature.moments, huMoments);
                        for (const auto &hu : huMoments) {
                                featureVector.push_back(hu); // Add each Hu1, Hu2, ..., Hu7 Hu Moment to the feature vector
                        }
                        // For cv::Point (centroid), add its x and y coordinates
                        featureVector.push_back(static_cast<double>(feature.centroid.x));
                        featureVector.push_back(static_cast<double>(feature.centroid.y));
                        featureVector.push_back(feature.percentFilled);
                        featureVector.push_back(feature.aspectRatio);
                        featureVector.push_back(feature.orientation);
                }
               
                std::cout << std::endl;
                std::string classifiedLabel = classifyNewFeatureVector(featureVector, knownObjects);
                displayLabelOnStream(frame, classifiedLabel); 
                cv::imshow("Baseline system ", frame);


                //Task 9:
                // std::string classifiedLabelKNN = classifyNewFeatureVectorKNN(featureVector, knn);
                // std::cout << "Classified Label: " << classifiedLabelKNN << std::endl;

                // displayLabelOnStream(frame, classifiedLabelKNN);
                // cv::imshow("KNN Classification", frame);

                }

        delete capdev;
        return(0);
}



    
