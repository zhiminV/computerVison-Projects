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

    
    while(true) {
        *capdev >> frame; // get a new frame from camera
        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        } 

        DetectAndExtractTargetCorners(frame);

        imshow("Video", frame);
        

        char key = cv::waitKey(10);
        if( key == 'q') {
            break;
        }
    
        else if(key == 's'){
            
        }
    }

delete capdev;
return(0);

}



    
