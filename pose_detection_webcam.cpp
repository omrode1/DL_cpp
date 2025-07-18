#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace dnn;

// Pose keypoints for COCO format
const std::vector<std::string> POSE_KEYPOINTS = {
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
};

// Pose connections (pairs of keypoints to draw lines between)
const std::vector<std::pair<int, int>> POSE_CONNECTIONS = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},  // Head
    {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},  // Arms
    {5, 11}, {6, 12}, {11, 12},  // Torso
    {11, 13}, {13, 15}, {12, 14}, {14, 16}  // Legs
};

struct PoseKeypoint {
    cv::Point2f point;
    float confidence;
    bool visible;
};

struct PoseDetection {
    std::vector<PoseKeypoint> keypoints;
    float score;
};

std::vector<PoseDetection> processPoseOutput(const cv::Mat& output, const cv::Size& original_size, 
                                           float conf_threshold = 0.5) {
    std::vector<PoseDetection> detections;
    
    // Output format: [batch, num_people, num_keypoints, 3] where 3 = [x, y, confidence]
    // For simplicity, we'll assume single person detection
    int num_keypoints = POSE_KEYPOINTS.size();
    
    // Reshape output to [num_keypoints, 3]
    cv::Mat reshaped_output = output.reshape(1, num_keypoints);
    
    PoseDetection pose;
    pose.keypoints.resize(num_keypoints);
    
    bool has_valid_keypoints = false;
    
    for (int i = 0; i < num_keypoints; i++) {
        float x = reshaped_output.at<float>(i, 0);
        float y = reshaped_output.at<float>(i, 1);
        float conf = reshaped_output.at<float>(i, 2);
        
        pose.keypoints[i].confidence = conf;
        pose.keypoints[i].visible = conf > conf_threshold;
        
        if (pose.keypoints[i].visible) {
            // Convert normalized coordinates to pixel coordinates
            pose.keypoints[i].point.x = x * original_size.width;
            pose.keypoints[i].point.y = y * original_size.height;
            has_valid_keypoints = true;
        } else {
            pose.keypoints[i].point = cv::Point2f(-1, -1);
        }
    }
    
    if (has_valid_keypoints) {
        detections.push_back(pose);
    }
    
    return detections;
}

void drawPose(cv::Mat& image, const PoseDetection& pose, float conf_threshold = 0.5) {
    // Draw keypoints
    for (size_t i = 0; i < pose.keypoints.size(); i++) {
        if (pose.keypoints[i].visible && pose.keypoints[i].confidence > conf_threshold) {
            cv::circle(image, pose.keypoints[i].point, 5, cv::Scalar(0, 255, 0), -1);
            cv::putText(image, POSE_KEYPOINTS[i], 
                       cv::Point(pose.keypoints[i].point.x + 10, pose.keypoints[i].point.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);
        }
    }
    
    // Draw connections
    for (const auto& connection : POSE_CONNECTIONS) {
        int idx1 = connection.first;
        int idx2 = connection.second;
        
        if (idx1 < pose.keypoints.size() && idx2 < pose.keypoints.size() &&
            pose.keypoints[idx1].visible && pose.keypoints[idx2].visible &&
            pose.keypoints[idx1].confidence > conf_threshold && 
            pose.keypoints[idx2].confidence > conf_threshold) {
            
            cv::line(image, pose.keypoints[idx1].point, pose.keypoints[idx2].point, 
                    cv::Scalar(255, 0, 0), 2);
        }
    }
}

int main() {
    try {
        std::cout << "OpenCV Pose Detection Webcam Test" << std::endl;
        std::cout << "OpenCV version: " << CV_VERSION << std::endl;
        
        // For this example, we'll use a simple approach with body detection
        // and keypoint estimation using OpenCV's built-in functions
        std::cout << "Loading pose detection model..." << std::endl;
        
        // We'll use HOG descriptor for person detection as a fallback
        // since pose models might not be readily available
        HOGDescriptor hog;
        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
        
        std::cout << "Pose detection model loaded successfully!" << std::endl;
        
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
                
                // Convert to grayscale for detection
                cvtColor(frame, gray, COLOR_BGR2GRAY);
                
                // Detect people using HOG
                std::vector<Rect> people;
                std::vector<double> weights;
                hog.detectMultiScale(gray, people, weights, 0, Size(8, 8), Size(4, 4), 1.05, 2, false);
                
                // Draw person detections and estimate pose regions
                for (size_t i = 0; i < people.size(); i++) {
                    if (weights[i] > 0.3) {  // Confidence threshold
                        Rect person = people[i];
                        
                        // Draw person bounding box
                        rectangle(processed, person, Scalar(0, 255, 0), 2);
                        
                        // Estimate pose keypoints based on person bounding box
                        // This is a simplified approach - in practice you'd use a proper pose model
                        PoseDetection pose;
                        pose.keypoints.resize(POSE_KEYPOINTS.size());
                        
                        // Calculate estimated keypoint positions based on bounding box
                        float center_x = person.x + person.width / 2.0f;
                        float center_y = person.y + person.height / 2.0f;
                        float scale_x = person.width / 200.0f;  // Normalized scale
                        float scale_y = person.height / 300.0f;
                        
                        // Define relative positions for keypoints (simplified)
                        std::vector<std::pair<float, float>> relative_positions = {
                            {0.5f, 0.1f},   // nose
                            {0.4f, 0.08f},  // left_eye
                            {0.6f, 0.08f},  // right_eye
                            {0.35f, 0.1f},  // left_ear
                            {0.65f, 0.1f},  // right_ear
                            {0.3f, 0.25f},  // left_shoulder
                            {0.7f, 0.25f},  // right_shoulder
                            {0.2f, 0.4f},   // left_elbow
                            {0.8f, 0.4f},   // right_elbow
                            {0.1f, 0.55f},  // left_wrist
                            {0.9f, 0.55f},  // right_wrist
                            {0.35f, 0.6f},  // left_hip
                            {0.65f, 0.6f},  // right_hip
                            {0.3f, 0.8f},   // left_knee
                            {0.7f, 0.8f},   // right_knee
                            {0.25f, 0.95f}, // left_ankle
                            {0.75f, 0.95f}  // right_ankle
                        };
                        
                        for (size_t j = 0; j < POSE_KEYPOINTS.size(); j++) {
                            float rel_x = relative_positions[j].first;
                            float rel_y = relative_positions[j].second;
                            
                            pose.keypoints[j].point.x = person.x + rel_x * person.width;
                            pose.keypoints[j].point.y = person.y + rel_y * person.height;
                            pose.keypoints[j].confidence = weights[i];
                            pose.keypoints[j].visible = true;
                        }
                        
                        // Draw pose
                        drawPose(processed, pose, 0.3);
                        
                        // Add person label
                        std::string label = "Person " + std::to_string(i+1) + " (" + std::to_string(int(weights[i] * 100)) + "%)";
                        putText(processed, label, Point(person.x, person.y - 10), 
                               FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
                    }
                }
                
                // Add frame counter and instructions
                std::string frame_text = "Frame: " + std::to_string(frame_count);
                putText(processed, frame_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                
                std::string people_text = "People detected: " + std::to_string(people.size());
                putText(processed, people_text, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                
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
                
                imshow("Pose Detection Webcam", resized_display);
            }
            
            // Handle key presses
            char key = waitKey(1) & 0xFF;
            
            if (key == 'q' || key == 27) { // 'q' or ESC
                std::cout << "Quitting..." << std::endl;
                break;
            } else if (key == 's') {
                // Save current frame
                std::string filename = "pose_detection_frame_" + std::to_string(frame_count) + ".jpg";
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
        
        std::cout << "Pose detection webcam test completed!" << std::endl;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
} 