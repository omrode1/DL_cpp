#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace dnn;

// COCO keypoints
const std::vector<std::string> POSE_KEYPOINTS = {
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"
};

// COCO pose pairs
const int POSE_PAIRS[20][2] = {
    {1,2}, {1,5}, {2,3}, {3,4}, {5,6}, {6,7}, {1,8}, {8,9}, {9,10}, {1,11}, {11,12}, {12,13}, {1,0}, {0,14}, {14,16}, {0,15}, {15,17}
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

void drawPose(cv::Mat& image, const PoseDetection& pose, float conf_threshold = 0.3) {
    // Draw keypoints
    for (size_t i = 0; i < pose.keypoints.size(); i++) {
        const PoseKeypoint& kp = pose.keypoints[i];
        
        if (kp.visible && kp.confidence > conf_threshold) {
            cv::circle(image, kp.point, 5, cv::Scalar(0, 255, 0), -1);
            
            if (i < 5) { // Label first few keypoints
                cv::putText(image, POSE_KEYPOINTS[i], 
                           cv::Point(kp.point.x + 5, kp.point.y - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);
            }
        }
    }
    
    // Draw pose connections
    for (int i = 0; i < 20; i++) {
        int idx1 = POSE_PAIRS[i][0];
        int idx2 = POSE_PAIRS[i][1];
        
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
        std::cout << "Real OpenPose Detection Webcam Test" << std::endl;
        std::cout << "OpenCV version: " << CV_VERSION << std::endl;
        
        // Load OpenPose model
        std::string model_path = "pose_coco.caffemodel";
        std::string config_path = "pose_coco.prototxt";
        
        std::cout << "Loading OpenPose model..." << std::endl;
        Net net = readNetFromCaffe(config_path, model_path);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
        std::cout << "OpenPose model loaded successfully!" << std::endl;
        
        // Open webcam
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
                Mat processed = frame.clone();
                
                // Prepare input for OpenPose
                Mat input_blob = blobFromImage(frame, 1.0/255.0, Size(368, 368), Scalar(0, 0, 0), false, false);
                
                // Run inference
                net.setInput(input_blob);
                Mat output = net.forward();
                
                // Process output (simplified)
                // OpenPose output: [batch, num_keypoints + num_pafs, height, width]
                int num_keypoints = 18;
                
                // Extract keypoint heatmaps
                std::vector<cv::Point2f> keypoints;
                keypoints.resize(num_keypoints);
                
                for (int i = 0; i < num_keypoints; i++) {
                    Mat heatmap(output.size[2], output.size[3], CV_32F, output.ptr<float>(0, i));
                    
                    // Find maximum in heatmap
                    double min_val, max_val;
                    cv::Point min_loc, max_loc;
                    cv::minMaxLoc(heatmap, &min_val, &max_val, &min_loc, &max_loc);
                    
                    if (max_val > 0.1) {
                        // Convert to original image coordinates
                        float x = (float)max_loc.x * frame.cols / heatmap.cols;
                        float y = (float)max_loc.y * frame.rows / heatmap.rows;
                        keypoints[i] = cv::Point2f(x, y);
                    } else {
                        keypoints[i] = cv::Point2f(-1, -1);
                    }
                }
                
                // Create pose detection
                PoseDetection pose;
                pose.keypoints.resize(num_keypoints);
                
                for (int i = 0; i < num_keypoints; i++) {
                    pose.keypoints[i].point = keypoints[i];
                    pose.keypoints[i].visible = (keypoints[i].x >= 0 && keypoints[i].y >= 0);
                    pose.keypoints[i].confidence = 0.8f; // Placeholder
                }
                
                // Draw pose
                drawPose(processed, pose, 0.3);
                
                // Add info
                std::string frame_text = "Frame: " + std::to_string(frame_count);
                putText(processed, frame_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                
                putText(processed, "Press 'q' to quit, 's' to save, 'p' to pause", Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                
                imshow("Real OpenPose Detection", processed);
            }
            
            char key = waitKey(1) & 0xFF;
            
            if (key == 'q' || key == 27) {
                std::cout << "Quitting..." << std::endl;
                break;
            } else if (key == 's') {
                std::string filename = "real_pose_frame_" + std::to_string(frame_count) + ".jpg";
                imwrite(filename, frame);
                std::cout << "Saved frame as " << filename << std::endl;
            } else if (key == 'p') {
                paused = !paused;
                std::cout << (paused ? "Paused" : "Resumed") << std::endl;
            }
        }
        
        cap.release();
        destroyAllWindows();
        std::cout << "Real OpenPose detection completed!" << std::endl;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
} 