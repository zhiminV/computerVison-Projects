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
#include "calibrate.h"
#include <string>


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

    cv::namedWindow("Detect Corners", 1); // identifies a window
    cv::Mat frame;
    std::vector<cv::Point2f> corners;
    std::vector<cv::Mat> calibration_images;
    cv::Size patternSize(9, 6); // checkerboard has 9 columns and 6 rows of internal corners
    std::vector<cv::Mat> rvecs, tvecs;
    double initialMatrix[] = {1, 0, (double) frame.cols/2, 0, 1, (double) frame.rows/2, 0, 0, 1};
    cv::Mat camera_matrix(cv::Size(3,3), CV_64FC1, &initialMatrix);
    cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64FC1); // Assuming 5 distortion coefficients


    
    while(true) {
        *capdev >> frame; // get a new frame from camera
        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        } 

        DetectAndExtractTargetCorners(frame,corners,patternSize);

        imshow("Video", frame);

        char key = cv::waitKey(10);
        if( key == 'q') {
            break;
        }
    
        else if(key == 's'){
            saveCalibrationData(corners,patternSize); // Save corners from the last detection
            // Save the image
            calibration_images.push_back(frame.clone());
            imwrite("calibration_image_" + to_string(corner_list.size()) + ".jpg", frame);

            // Check for enough calibration frames
            if(corner_list.size() >= 5) {

                // Print initial camera matrix and distortion coefficients
                std::cout << "Initial Camera Matrix:\n" << camera_matrix << std::endl;
                std::cout << "Initial Distortion Coefficients:\n" << dist_coeffs << std::endl;

                // Calibrate the camera
                double  re_projection_error = cv::calibrateCamera(point_list, corner_list, frame.size(), camera_matrix, dist_coeffs, rvecs, tvecs, cv::CALIB_FIX_ASPECT_RATIO);

                std::cout << "Calibrated Camera Matrix:\n" << camera_matrix << std::endl;
                std::cout << "Calibrated Distortion Coefficients:\n" << dist_coeffs << std::endl;
                std::cout << "Re-projection Error: " << re_projection_error << std::endl;

                saveCalibrationParameters(camera_matrix, dist_coeffs, "calibration.csv");
            }
            else {
                std::cout << "Not enough calibration frames. Please capture 5 frames." << std::endl;
            }
            
            
        } 
        
    }

// delete capdev;
return(0);

}



    