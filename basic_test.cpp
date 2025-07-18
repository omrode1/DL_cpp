#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace dnn;

int main() {
    try {
        std::cout << "OpenCV DNN Basic Test" << std::endl;
        std::cout << "OpenCV version: " << CV_VERSION << std::endl;
        
        // Load image
        std::cout << "Loading image..." << std::endl;
        Mat img = imread("image.jpg");
        if (img.empty()) {
            std::cerr << "Error: Could not load image.jpg" << std::endl;
            return -1;
        }
        
        std::cout << "Image loaded successfully!" << std::endl;
        std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;
        
        // Display the original image
        imshow("Original Image", img);
        std::cout << "Showing original image. Press any key to continue..." << std::endl;
        waitKey(0);
        
        // Demonstrate basic image processing
        Mat gray, blurred, edges;
        
        // Convert to grayscale
        cvtColor(img, gray, COLOR_BGR2GRAY);
        imshow("Grayscale", gray);
        std::cout << "Showing grayscale image. Press any key to continue..." << std::endl;
        waitKey(0);
        
        // Apply Gaussian blur
        GaussianBlur(gray, blurred, Size(15, 15), 0);
        imshow("Blurred", blurred);
        std::cout << "Showing blurred image. Press any key to continue..." << std::endl;
        waitKey(0);
        
        // Detect edges
        Canny(blurred, edges, 50, 150);
        imshow("Edge Detection", edges);
        std::cout << "Showing edge detection. Press any key to continue..." << std::endl;
        waitKey(0);
        
        // Demonstrate DNN blob creation (without running inference)
        std::cout << "Demonstrating DNN blob creation..." << std::endl;
        Mat blob = blobFromImage(img, 1.0/255.0, Size(224, 224), Scalar(0, 0, 0), true, false);
        std::cout << "Blob created successfully!" << std::endl;
        std::cout << "Blob size: " << blob.size[0] << "x" << blob.size[1] << "x" << blob.size[2] << "x" << blob.size[3] << std::endl;
        
        // Show final result with all processing
        Mat result;
        hconcat(img, gray, result);
        Mat temp;
        cvtColor(blurred, temp, COLOR_GRAY2BGR);
        hconcat(result, temp, result);
        cvtColor(edges, temp, COLOR_GRAY2BGR);
        hconcat(result, temp, result);
        
        // Resize to fit screen
        Mat display;
        resize(result, display, Size(), 0.5, 0.5);
        
        imshow("All Processing Results", display);
        std::cout << "Showing all processing results. Press any key to exit..." << std::endl;
        waitKey(0);
        
        std::cout << "Test completed successfully!" << std::endl;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
} 