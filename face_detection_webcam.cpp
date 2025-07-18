#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace dnn;

int main() {
    try {
        std::cout << "OpenCV Face Detection Webcam Test" << std::endl;
        std::cout << "OpenCV version: " << CV_VERSION << std::endl;
        
        // Load face detection cascade
        std::cout << "Loading face detection model..." << std::endl;
        CascadeClassifier face_cascade;
        if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml")) {
            std::cerr << "Error: Could not load face cascade!" << std::endl;
            std::cout << "Trying alternative path..." << std::endl;
            if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
                std::cerr << "Error: Could not load face cascade from alternative path!" << std::endl;
                return -1;
            }
        }
        std::cout << "Face detection model loaded successfully!" << std::endl;
        
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
                Mat gray;
                
                // Convert to grayscale for face detection
                cvtColor(frame, gray, COLOR_BGR2GRAY);
                
                // Detect faces
                std::vector<Rect> faces;
                face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
                
                // Draw face detection results
                for (size_t i = 0; i < faces.size(); i++) {
                    Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
                    ellipse(processed, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(255, 0, 255), 4);
                    
                    // Add face number
                    putText(processed, "Face " + std::to_string(i+1), 
                           Point(faces[i].x, faces[i].y - 10), 
                           FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 255), 2);
                }
                
                // Add frame counter and instructions
                std::string frame_text = "Frame: " + std::to_string(frame_count);
                putText(processed, frame_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                
                std::string face_text = "Faces detected: " + std::to_string(faces.size());
                putText(processed, face_text, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                
                putText(processed, "Press 'q' to quit, 's' to save, 'p' to pause", Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                
                // Create a combined display
                Mat display;
                Mat gray_bgr;
                cvtColor(gray, gray_bgr, COLOR_GRAY2BGR);
                
                // Stack images horizontally
                hconcat(processed, gray_bgr, display);
                
                // Resize to fit screen
                Mat resized_display;
                resize(display, resized_display, Size(), 0.8, 0.8);
                
                imshow("Face Detection Webcam", resized_display);
            }
            
            // Handle key presses
            char key = waitKey(1) & 0xFF;
            
            if (key == 'q' || key == 27) { // 'q' or ESC
                std::cout << "Quitting..." << std::endl;
                break;
            } else if (key == 's') {
                // Save current frame
                std::string filename = "face_detection_frame_" + std::to_string(frame_count) + ".jpg";
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
        
        std::cout << "Face detection webcam test completed!" << std::endl;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
} 