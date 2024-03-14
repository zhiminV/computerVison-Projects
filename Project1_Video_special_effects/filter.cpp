/*
Zhimin Liang
Spring 2024
CS5330 Project 1

Purpose: This file implements various simple filters for image processing using OpenCV. 
*/


#include <opencv2/opencv.hpp>
#include <cstdio>
#include "filter.h"
#include "faceDetect.h"
/* 
Task 4: Display alternative greyscale live video
Converts the input color image to an alternative grayscale by subtracting the red channel from 255.
Arguments:
  cv::Mat &src - Input image.
cv::Mat &dst - Output grayscale image.
Returns:
  0 on successful execution.
*/
int greyscale(cv::Mat &src, cv::Mat &dst) {
    // check whether the input is in the correct format
    if (src.channels() != 3) {
        return -1; 
    }

    // Subtract the red channel from 255 and copy the value to all three color channels
    cv::Mat transformed = cv::Scalar::all(255) - src;

    // Convert the transformed image to grayscale
    cv::cvtColor(transformed, dst, cv::COLOR_BGR2GRAY);

    return 0;
}

/*
Task 5: Implement a Sepia tone filter
Applies a Sepia tone effect to the input color image.
Arguments:
  cv::Mat &src - Input image.
  cv::Mat &dst - Output image with Sepia tone effect.
Returns:
  0 on successful execution.
*/
int sepiaTone(cv::Mat &src, cv::Mat &dst) {

    cv::Mat sepiaMatrix = (cv::Mat_<float>(3, 3) << 0.272, 0.534, 0.131,  // Blue coefficients
                                                    0.349, 0.686, 0.168,  // Green coefficients
                                                    0.393, 0.769, 0.189); // Red coefficients


    // Allocate dst to be an image of the same size and type as src
    dst.create(src.size(), src.type());

    // Apply Sepia tone filter 
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b color = src.at<cv::Vec3b>(y, x);

            //Calculate new color values using the sepiaMatrix
            //newRed = 0.272 * R + 0.534 * G + 0.131 * B
            float newBlue = sepiaMatrix.at<float>(0, 0) * color[0] + sepiaMatrix.at<float>(0, 1) * color[1] + sepiaMatrix.at<float>(0, 2) * color[2];
            float newGreen = sepiaMatrix.at<float>(1, 0) * color[0] + sepiaMatrix.at<float>(1, 1) * color[1] + sepiaMatrix.at<float>(1, 2) * color[2];
            float newRed = sepiaMatrix.at<float>(2, 0) * color[0] + sepiaMatrix.at<float>(2, 1) * color[1] + sepiaMatrix.at<float>(2, 2) * color[2];

        
            newBlue = std::min(255.0f, std::max(0.0f, newBlue));
            newGreen = std::min(255.0f, std::max(0.0f, newGreen));
            newRed = std::min(255.0f, std::max(0.0f, newRed));
            //Sets the new color values in the destination image.
           dst.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<uchar>(newBlue), static_cast<uchar>(newGreen), static_cast<uchar>(newRed));
        }
    }
    return (0);
}

/*
Applies a Sepia tone effect with a Vignette effect to the input color image.
Arguments:
  cv::Mat &src - Input image.
  cv::Mat &dst - Output image with Sepia tone and Vignette effects.
Returns:
  0 on successful execution.
*/
int sepiawithVignette(cv::Mat &src, cv::Mat &dst) {

    cv::Mat sepiaMatrix = (cv::Mat_<float>(3, 3) << 0.272, 0.534, 0.131,  // Blue coefficients
                                                    0.349, 0.686, 0.168,  // Green coefficients
                                                    0.393, 0.769, 0.189); // Red coefficients


    // Allocate dst to be an image of the same size and type as src
    dst.create(src.size(), src.type());

    // Apply Sepia tone filter 
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b color = src.at<cv::Vec3b>(y, x);

            //Calculate new color values using the sepiaMatrix
            //newBlue = 0.272 * R + 0.534 * G + 0.131 * B
            float newBlue = sepiaMatrix.at<float>(0, 0) * color[0] + sepiaMatrix.at<float>(0, 1) * color[1] + sepiaMatrix.at<float>(0, 2) * color[2];
            float newGreen = sepiaMatrix.at<float>(1, 0) * color[0] + sepiaMatrix.at<float>(1, 1) * color[1] + sepiaMatrix.at<float>(1, 2) * color[2];
            float newRed = sepiaMatrix.at<float>(2, 0) * color[0] + sepiaMatrix.at<float>(2, 1) * color[1] + sepiaMatrix.at<float>(2, 2) * color[2];

            // Apply vignette effect (darker towards the edges)
            float vignette = 1.0 - 0.3 * (std::sqrt((x - src.cols / 2.0) * (x - src.cols / 2.0) + (y - src.rows / 2.0) * (y - src.rows / 2.0)) / (std::min(src.cols, src.rows) / 2.0));

            newBlue *= vignette;
            newGreen *= vignette;
            newRed *= vignette;
           

            // Clamp the values to the valid range [0, 255]
            newBlue = std::min(255.0f, std::max(0.0f, newBlue));
            newGreen = std::min(255.0f, std::max(0.0f, newGreen));
            newRed = std::min(255.0f, std::max(0.0f, newRed));

            //Sets the new color values in the destination image.
           dst.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<uchar>(newBlue), static_cast<uchar>(newGreen), static_cast<uchar>(newRed));
        }
    }
    return (0);
}
/*
Task 6A: Implement a 5x5 blur filter using the at method
Applies a 5x5 blur filter to the input image using the at method.
Arguments:
  cv::Mat &src - Input image.
  cv::Mat &dst - Output image with applied blur.
Returns:
  0 on successful execution.
*/

/* 5x5_filters =[1 2 4 2 1; 
                2 4 8 4 2; 
                4 8 16 8 4; 
                2 4 8 4 2; 
                1 2 4 2 1]
*/
int blur5x5_1( cv::Mat &src, cv::Mat &dst ){//pass image by reference
    dst.create(src.size(), src.type());
    src.copyTo(dst);

    for(int i=1;i<src.rows-2;i++){
        for(int j=1;j<src.cols-2;j++){
            for(int k=0;k<src.channels();k++){// color channel
                int sum = src.at<cv::Vec3b>(i-1,j-1)[k] + 2 * src.at<cv::Vec3b>(i-1,j)[k] + 4 * src.at<cv::Vec3b>(i-1,j+1)[k]
                         + 2 * src.at<cv::Vec3b>(i,j-1)[k] + 4 * src.at<cv::Vec3b>(i,j)[k] + 8 * src.at<cv::Vec3b>(i,j+1)[k]
                         + 4 * src.at<cv::Vec3b>(i+1,j-1)[k] + 8 * src.at<cv::Vec3b>(i+1,j)[k] + 16 * src.at<cv::Vec3b>(i+1,j+1)[k];

                //1+2+4+8+16+8+4+2+1 = 64
                // normalize value back to a range of [0,255]
                sum /= 64;

                dst.at<cv::Vec3b>(i,j)[k] = sum;
            }
        }
    }

    return (0);

}


/*
Task 6B: Implement a 5x5 blur filter, make it faster using the pointer method
Applies a 5x5 blur filter to the input image using the pointer method for faster processing.
Arguments:
  cv::Mat &src - Input image.
  cv::Mat &dst - Output image with applied blur.
Returns:
  0 on successful execution.
*/
/*  1x5_filters = [1 2 4 2 1] */
int blur5x5_2( cv::Mat &src, cv::Mat &dst ){
    dst.create(src.size(), src.type());
    src.copyTo(dst);
    int kernel[5] = {1, 2, 4, 2, 1};

    for (int i = 2; i < src.rows - 2; i++) {
        cv::Vec3b *ptrmd = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 2; j < src.cols - 2; j++) {
            for (int k = 0; k < src.channels(); k++) {
                int sum = 0;
                for (int m = 0; m < 5; m++) {
                    sum += kernel[m] * ptrmd[j - 2 + m][k];
                }
                // Normalize the value back to a range of [0, 255]
                sum /= 10;
                dptr[j][k] = sum;
            }
        }
    }

    //1x5 filter vertical
    for (int i = 2; i < src.rows - 2; i++) {
    cv::Vec3b *ptrmd = dst.ptr<cv::Vec3b>(i);
    cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 2; j < src.cols - 2; j++) {
            for (int k = 0; k < src.channels(); k++) {
                int sum = 0;
                for (int m = 0; m < 5; m++) {
                    sum += kernel[m] * ptrmd[j - 2 + m][k];
                }
                // Normalize the value back to a range of [0, 255]
                sum /= 10;
                dptr[j][k] = sum;
            }
        }
    }


    return (0);

}

/*
Task 7: Implement a 3x3 Sobel X and 3x3 Sobel Y filter as separable 1x3 filters
Applies separable 3x3 Sobel X and 3x3 Sobel Y filters to the input image.
Arguments:
  cv::Mat &src - Input image.
  cv::Mat &dst - Output image with applied Sobel filters.
Returns:
  0 on successful execution.
*/

/*-1  0  1
-2  0  2
-1  0  1*/
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), CV_16SC3);
    std::cout << "sobelX3x3: src.size() = " << src.size() << ", dst.size() = " << dst.size() << std::endl;
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
Task 7: Implement a 3x3 Sobel Y filter as separable 1x3 filters
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
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), CV_16SC3);
    std::cout << "sobelY3x3: src.size() = " << src.size() << ", dst.size() = " << dst.size() << std::endl;

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
Task 8: Implement a function that generates a gradient magnitude image from the X and Y Sobel images
Generates a gradient magnitude image from the input Sobel X and Y images.
Arguments:
  cv::Mat &sx - Input Sobel X image.
  cv::Mat &sy - Input Sobel Y image.
  cv::Mat &dst - Output gradient magnitude image.
Returns:
  0 on successful execution.
*/
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
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



/*
Task 9: Implement a function that blurs and quantizes a color image
Blurs and quantizes a color image to the specified number of levels.
Arguments:
  cv::Mat &src - Input image.
  cv::Mat &dst - Output blurred and quantized image.
  int levels - Number of quantization levels (default: 10).
Returns:
  0 on successful execution.
*/
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels = 10) {
    CV_Assert(levels > 0);

    // Create a temporary image for blurring
     blur5x5_2(src, dst);

    dst.create(src.size(), src.type());

    // Calculate the size of a bucket
    double bucketSize = 255.0 / levels;

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            for (int c = 0; c < src.channels(); c++) {
                // Quantize the color values
                double xt = src.at<cv::Vec3b>(y, x)[c] / bucketSize;
                double xf = round(xt) * bucketSize;

                // Apply the quantized value to the destination image
                dst.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(xf);
            }
        }
    }

    return 0;
}

/*
Task 11: Blur the image outside found faces
Blurs the regions outside the detected faces in the input image.
Arguments:
  cv::Mat &src - Input image.
  cv::Mat &dst - Output image with blurred background.
Returns:
  0 on successful execution.
*/
int blurOutsideFaces(cv::Mat &src, cv::Mat &dst) {
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);

    // Detect faces
    std::vector<cv::Rect> faces;
    detectFaces(grey, faces);

    // Create a mask to blur outside faces
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8U);
    for (const auto &face : faces) {
        mask(face) = 255;
    }

    // Apply blur only outside faces
    cv::Mat blurred;
    cv::blur(src, blurred, cv::Size(15, 15));  // Adjust kernel size as needed

    // Combine the original faces with the blurred background
    cv::bitwise_and(src, src, dst, mask);
    cv::bitwise_and(blurred, blurred, blurred, ~mask);
    cv::add(dst, blurred, dst);

    return 0;
}
/*
Task 11: Embossing effect using Sobel X and Sobel Y
Applies an embossing effect to the input image using Sobel X and Sobel Y filters.
Arguments:
  cv::Mat &src - Input image.
  cv::Mat &dst - Output image with embossing effect.
Returns:
  0 on successful execution.
*/
int embossingEffect(cv::Mat &src, cv::Mat &dst) {
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);

    cv::Mat sobelX, sobelY;
    cv::Sobel(grey, sobelX, CV_16S, 1, 0);
    cv::Sobel(grey, sobelY, CV_16S, 0, 1);

    cv::Mat embossing;
    cv::multiply(sobelX, 0.7071, sobelX);
    cv::multiply(sobelY, 0.7071, sobelY);
    cv::add(sobelX, sobelY, embossing);

    cv::convertScaleAbs(embossing, dst);

    return 0;
}


/*
Task11: Add Sparkles Above Detected Faces
Adds sparkles above the detected faces in the input frame.
Arguments:
  cv::Mat &frame - Input frame.
  const std::vector<cv::Rect> &faces - Vector containing the rectangles of detected faces.
Returns:
  None (void function).
*/
void addSparklesAboveFaces(cv::Mat &frame, const std::vector<cv::Rect> &faces) {
    // Define the sparkle color 
    cv::Scalar sparkleColor(255, 255, 255);  // White sparkles

    // Define the size and intensity of the sparkles
    int sparkleSize = 5;
    int sparkleIntensity = 150;

    // Iterate over each detected face
    for (const auto &face : faces) {
        // Calculate the position for the halo above the face
        int haloY = std::max(0, face.y - 30);  // Adjust the distance from the face

        // Add sparkles to the halo
        for (int i = 0; i < sparkleIntensity; ++i) {
            int sparkleX = face.x + rand() % face.width;  // Random X position within the face
            int sparkleY = haloY + rand() % 30;           // Random Y position in the halo

            // Draw the sparkle as a filled circle
            cv::circle(frame, cv::Point(sparkleX, sparkleY), sparkleSize, sparkleColor, -1);
        }
    }
}

/*
Task: Apply Cartoon Effect
Applies a cartoon effect to the input image by combining bilateral filtering for smoothing and adaptive thresholding for edge detection.
Arguments:
  cv::Mat &input - Input image.
  cv::Mat &output - Output image with a cartoon effect.
Returns:
  None (void function).
*/
void cartoonize(cv::Mat &input, cv::Mat &output) {
  // Convert the input image to grayscale
  cv::Mat gray;
  cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);

  // Apply bilateral filter to smooth the image while preserving edges
  cv::Mat smooth;
  cv::bilateralFilter(input, smooth, 9, 75, 75);

  // Apply edge detection
  cv::Mat edges;
  cv::adaptiveThreshold(gray, edges, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 9, 15);

  // Combine the smoothed image with the edges to create a cartoon effect
  cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
  cv::bitwise_and(smooth, edges, output);
}
