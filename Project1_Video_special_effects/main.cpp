#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <cstdlib>

int main() {
    // Load two different texture images
    cv::Mat texture1 = cv::imread("/Users/lzm/Desktop/texture1.jpg");
    cv::Mat texture2 = cv::imread("/Users/lzm/Desktop/texture2.jpg");

    // Check if images are loaded successfully
    if (texture1.empty() || texture2.empty()) {
        std::cerr << "Error loading images." << std::endl;
        return -1;
    }

    // Apply Sobel filters to obtain gradient images
    cv::Mat sobelX1, sobelY1, gradient1;
    sobelX3x3(texture1, sobelX1);
    sobelY3x3(texture1, sobelY1);
    magnitude(sobelX1, sobelY1, gradient1);

    cv::Mat sobelX2, sobelY2, gradient2;
    sobelX3x3(texture2, sobelX2);
    sobelY3x3(texture2, sobelY2);
    magnitude(sobelX2, sobelY2, gradient2);

    // Display the original images and gradient magnitudes
    cv::imshow("Texture 1", texture1);
    cv::imshow("Gradient Magnitude 1", gradient1);

    cv::imshow("Texture 2", texture2);
    cv::imshow("Gradient Magnitude 2", gradient2);

    // Calculate average energy of the gradient magnitude
    double avgEnergy1 = cv::mean(gradient1.mul(gradient1))[0];
    double avgEnergy2 = cv::mean(gradient2.mul(gradient2))[0];

    std::cout << "Average Energy of Gradient Magnitude for Texture 1: " << avgEnergy1 << std::endl;
    std::cout << "Average Energy of Gradient Magnitude for Texture 2: " << avgEnergy2 << std::endl;

    // Wait for key press and close the windows
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}