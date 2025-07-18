#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace dnn;

int main() {
    try {
        std::cout << "OpenCV Webcam Test" << std::endl;
        std::cout << "OpenCV version: " << CV_VERSION << std::endl;
        
        // Open webcam (device 1 as requested)
        std::cout << "Opening webcam at device 1..." << std::endl;
        VideoCapture cap(1);
        
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open webcam at device 1. Trying device 0..." << std::endl;
            cap.open(0);
            if (!cap.isOpened()) {
                std::cerr << "Error: Could not open any webcam!" << std::endl;
                return -1;
            }
        }
        
        std::cout << "Webcam opened successfully!" << std::endl;
        
        // Set webcam properties
        cap.set(CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(CAP_PROP_FPS, 30);
        
        std::cout << "Press 'q' to quit, 's' to save current frame, 'p' to pause/resume" << std::endl;
        
        bool paused = false;
        int frame_count = 0;
        Mat frame;
        
        while (true) {
            if (!paused) {
                cap >> frame;
                
                if (frame.empty()) {
                    std::cerr << "Error: Could not read frame from webcam!" << std::endl;
                    break;
                }
                
                frame_count++;
                
                // Create a copy for processing
                Mat processed = frame.clone();
                
                // Apply some basic processing
                Mat gray, blurred, edges;
                
                // Convert to grayscale
                cvtColor(frame, gray, COLOR_BGR2GRAY);
                
                // Apply Gaussian blur
                GaussianBlur(gray, blurred, Size(15, 15), 0);
                
                // Detect edges
                Canny(blurred, edges, 50, 150);
                
                // Add frame counter and instructions
                std::string frame_text = "Frame: " + std::to_string(frame_count);
                putText(processed, frame_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                putText(processed, "Press 'q' to quit, 's' to save, 'p' to pause", Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                
                // Create a combined display
                Mat display;
                Mat gray_bgr, edges_bgr;
                cvtColor(gray, gray_bgr, COLOR_GRAY2BGR);
                cvtColor(edges, edges_bgr, COLOR_GRAY2BGR);
                
                // Stack images vertically
                vconcat(processed, gray_bgr, display);
                Mat temp;
                vconcat(display, edges_bgr, display);
                
                // Resize to fit screen
                Mat resized_display;
                resize(display, resized_display, Size(), 0.8, 0.8);
                
                imshow("Webcam Processing", resized_display);
            }
            
            // Handle key presses
            char key = waitKey(1) & 0xFF;
            
            if (key == 'q' || key == 27) { // 'q' or ESC
                std::cout << "Quitting..." << std::endl;
                break;
            } else if (key == 's') {
                // Save current frame
                std::string filename = "webcam_frame_" + std::to_string(frame_count) + ".jpg";
                imwrite(filename, frame);
                std::cout << "Saved frame as " << filename << std::endl;
            } else if (key == 'p') {
                paused = !paused;
                std::cout << (paused ? "Paused" : "Resumed") << std::endl;
            }
        }
        
        // Clean up
        cap.release();
        destroyAllWindows();
        
        std::cout << "Webcam test completed!" << std::endl;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
} 