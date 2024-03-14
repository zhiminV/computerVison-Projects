// /*
// Zhimin Liang
// Spring 2024
// CS5330 Project 1
// Purpose: Read an image from a file and display it
// */

// #include <opencv2/opencv.hpp>
// #include <cstdio>
// #include <cstring>
// #include <cstdlib>

// /*
// Function:
// Method:
// Return:
// */

// int main(int argc, char *argv[]){
//     //cv::Mat is used to represent matrices (images) in OpenCV. 
//     //It can store 2D or multi-dimensional arrays of pixel values.
//     //variable name src
//     cv::Mat src;
//     char filename[256];

//     //check if command line 
//     if(argc < 2){
//         printf("usage: %s <image filename> \n", argv[0]);
//         exit(-1);
//     }

//     strcpy(filename, argv[1]);

//     //read the image
//     src = cv::imread(filename);

//     if (src.data == NULL){
//         printf("error: unable to read image %s\n", filename);
//         exit(-1);
//     }

//     cv::namedWindow(filename,1);
//     cv::imshow(filename,src);

//     while(true){
//         printf("Type q to quit; Type m to modify the image\n");
//         // wait for a key press
//         int key = cv::waitKey(0);
//         switch (key){
//             // quit the program if 'q' is pressed
//             case 'q':
//                 exit(0);
//                 break;
//             // modify the image (swap the red and green channels)
//             case 'm':
//                 for (int i = 0; i < src.rows;i++){
//                     cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
//                     for (int j = 0; j<src.cols; j++){
//                         uchar temp = ptr[j][0];
//                         ptr[j][0] = ptr[j][1];
//                         ptr[j][1] = temp;
//                     }

//                 }
//                 cv::namedWindow("Swap RG",2);//cv::namedWindow(filename, flags)
//                 cv::imshow("Swap RG",src); //cv::imshow(window_name, image)
//                 break;

//             default:
//                 break;
//         }

//     }
    
//     return (0);

// }


