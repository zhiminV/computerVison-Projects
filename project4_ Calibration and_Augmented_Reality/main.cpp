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
            saveCalibrationData(corners); // Save corners from the last detection
            // Save the image
            calibration_images.push_back(frame.clone());
            imwrite("calibration_image_" + to_string(corner_list.size()) + ".jpg", frame);

            // Print corner coordinates and corresponding 3D world points
            cout << "Corner list contents:" << endl;
            for (int i = 0; i < corners.size(); ++i) {
                cout << "(" << corners[i].x << ", " << corners[i].y << ") ";
                cout << "World point: (" << point_list.back()[i][0] << ", " << point_list.back()[i][1] << ", " << point_list.back()[i][2] << ")";
                cout << endl;
            }
            
        } 
        
    }

delete capdev;
return(0);

}



    
