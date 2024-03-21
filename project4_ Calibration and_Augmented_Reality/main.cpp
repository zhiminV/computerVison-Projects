/*
Zhimin Liang
Spring 2024
CS5330 Project 3

Purpose: 
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "calibrate.h"

int main(int argc, char *argv[]) {
    cv::VideoCapture capdev(0);
    if (!capdev.isOpened()) {
        std::cerr << "Unable to open video device" << std::endl;
        return -1;
    }

    cv::Size refS(capdev.get(cv::CAP_PROP_FRAME_WIDTH), capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Expected size: " << refS.width << " " << refS.height << std::endl;

    cv::namedWindow("Detect Corners", 1);
    cv::Mat frame;
    std::vector<cv::Point2f> corners;
    std::vector<cv::Mat> calibration_images;
    cv::Size patternSize(9, 6);
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1, 0, refS.width / 2, 0, 1, refS.height / 2, 0, 0, 1);
    cv::Mat dist_coeffs = cv::Mat::zeros(0, 0, CV_64FC1);

    while (true) {
        capdev >> frame;
        if (frame.empty()) {
            std::cerr << "Frame is empty" << std::endl;
            break;
        }

        DetectAndExtractTargetCorners(frame, corners, patternSize);

        imshow("Video", frame);

        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        } else if (key == 's') {
            saveCalibrationData(corners, patternSize);
            calibration_images.push_back(frame.clone());
            imwrite("calibration_image_" + std::to_string(corner_list.size()) + ".jpg", frame);

            if (corner_list.size() >= 5) {
                std::cout << "Initial Camera Matrix:\n" << camera_matrix << std::endl;
                std::cout << "Initial Distortion Coefficients:\n" << dist_coeffs << std::endl;

                double re_projection_error = cv::calibrateCamera(point_list, corner_list, frame.size(), camera_matrix, dist_coeffs, rvecs, tvecs, cv::CALIB_FIX_ASPECT_RATIO);

                std::cout << "Calibrated Camera Matrix:\n" << camera_matrix << std::endl;
                std::cout << "Calibrated Distortion Coefficients:\n" << dist_coeffs << std::endl;
                std::cout << "Re-projection Error: " << re_projection_error << std::endl;

                saveCalibrationParameters(camera_matrix, dist_coeffs, "calibration.csv");
            } else {
                std::cout << "Not enough calibration frames. Please capture 5 frames." << std::endl;
            }
        }
    }

    return 0;
}
