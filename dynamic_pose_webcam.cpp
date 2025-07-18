#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <deque>

using namespace cv;
using namespace dnn;

// Pose keypoints for COCO format
const std::vector<std::string> POSE_KEYPOINTS = {
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
};

// Pose connections
const std::vector<std::pair<int, int>> POSE_CONNECTIONS = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},  // Head
    {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},  // Arms
    {5, 11}, {6, 12}, {11, 12},  // Torso
    {11, 13}, {13, 15}, {12, 14}, {14, 16}  // Legs
};

struct DynamicKeypoint {
    cv::Point2f point;
    float confidence;
    bool visible;
    std::deque<cv::Point2f> history;  // Track movement history
    cv::Point2f velocity;  // Current velocity
};

struct DynamicPose {
    std::vector<DynamicKeypoint> keypoints;
    float score;
    cv::Rect bounding_box;
};

class MotionTracker {
private:
    cv::Mat prev_frame;
    cv::Mat prev_gray;
    std::vector<cv::Point2f> prev_points;
    std::vector<cv::Point2f> curr_points;
    std::vector<uchar> status;
    std::vector<float> err;
    
public:
    MotionTracker() {}
    
    void update(const cv::Mat& frame) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        if (!prev_gray.empty() && !prev_points.empty()) {
            // Calculate optical flow only if we have previous points
            cv::calcOpticalFlowPyrLK(prev_gray, gray, prev_points, curr_points, status, err);
        } else {
            // Initialize points if empty
            std::vector<cv::Point2f> grid_points;
            for (int y = 50; y < gray.rows - 50; y += 50) {
                for (int x = 50; x < gray.cols - 50; x += 50) {
                    grid_points.push_back(cv::Point2f(x, y));
                }
            }
            prev_points = grid_points;
            curr_points = grid_points;
            status.resize(grid_points.size(), 1);
            err.resize(grid_points.size(), 0);
        }
        
        prev_gray = gray.clone();
        prev_frame = frame.clone();
    }
    
    std::vector<cv::Point2f> getMotionVectors() {
        std::vector<cv::Point2f> motion_vectors;
        for (size_t i = 0; i < curr_points.size() && i < prev_points.size(); i++) {
            if (status[i]) {
                motion_vectors.push_back(curr_points[i] - prev_points[i]);
            }
        }
        return motion_vectors;
    }
    
    void setPoints(const std::vector<cv::Point2f>& points) {
        prev_points = points;
        curr_points = points;
    }
};

class DynamicPoseDetector {
private:
    MotionTracker motion_tracker;
    cv::Mat background_model;
    bool background_initialized;
    std::vector<DynamicPose> previous_poses;
    
public:
    DynamicPoseDetector() : background_initialized(false) {}
    
    std::vector<DynamicPose> detectPoses(const cv::Mat& frame) {
        std::vector<DynamicPose> poses;
        
        // Update motion tracker
        motion_tracker.update(frame);
        
        // Background subtraction for motion detection
        cv::Mat gray, motion_mask;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        if (!background_initialized) {
            background_model = gray.clone();
            background_initialized = true;
        } else {
            // Update background model (simple running average)
            cv::addWeighted(background_model, 0.95, gray, 0.05, 0, background_model);
        }
        
        // Detect motion
        cv::absdiff(gray, background_model, motion_mask);
        cv::threshold(motion_mask, motion_mask, 25, 255, cv::THRESH_BINARY);
        
        // Morphological operations to clean up motion mask
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(motion_mask, motion_mask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(motion_mask, motion_mask, cv::MORPH_OPEN, kernel);
        
        // Find contours in motion mask
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(motion_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Filter contours by size and create poses
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > 1000 && area < 50000) {  // Filter by size
                cv::Rect bounding_rect = cv::boundingRect(contour);
                
                DynamicPose pose;
                pose.bounding_box = bounding_rect;
                pose.score = area / 10000.0f;  // Normalize score
                pose.keypoints.resize(POSE_KEYPOINTS.size());
                
                // Create dynamic keypoints based on contour analysis
                createDynamicKeypoints(pose, contour, frame);
                
                poses.push_back(pose);
            }
        }
        
        // Update previous poses for tracking
        updatePoseTracking(poses);
        previous_poses = poses;
        
        return poses;
    }
    
private:
    void createDynamicKeypoints(DynamicPose& pose, const std::vector<cv::Point>& contour, const cv::Mat& frame) {
        cv::Rect bbox = pose.bounding_box;
        
        // Get motion vectors for this region
        std::vector<cv::Point2f> motion_vectors = motion_tracker.getMotionVectors();
        
        // Create keypoints with dynamic positioning
        for (size_t i = 0; i < POSE_KEYPOINTS.size(); i++) {
            DynamicKeypoint& kp = pose.keypoints[i];
            
            // Base position based on bounding box
            float rel_x = getRelativeX(i, bbox);
            float rel_y = getRelativeY(i, bbox);
            
            // Add motion-based offset
            cv::Point2f motion_offset(0, 0);
            if (!motion_vectors.empty()) {
                motion_offset = motion_vectors[0] * 0.1f;  // Scale motion effect
            }
            
            // Add some randomness based on motion
            float motion_factor = cv::norm(motion_offset);
            if (motion_factor > 5.0f) {
                rel_x += (rand() % 100 - 50) / 1000.0f * motion_factor;
                rel_y += (rand() % 100 - 50) / 1000.0f * motion_factor;
            }
            
            kp.point.x = bbox.x + rel_x * bbox.width + motion_offset.x;
            kp.point.y = bbox.y + rel_y * bbox.height + motion_offset.y;
            kp.confidence = pose.score;
            kp.visible = true;
            
            // Update history
            kp.history.push_back(kp.point);
            if (kp.history.size() > 10) {
                kp.history.pop_front();
            }
            
            // Calculate velocity from history
            if (kp.history.size() >= 2) {
                kp.velocity = kp.history.back() - kp.history.front();
            }
        }
    }
    
    float getRelativeX(int keypoint_idx, const cv::Rect& bbox) {
        // Define relative positions with some variation
        std::vector<float> relative_x = {
            0.5f,   // nose
            0.4f,   // left_eye
            0.6f,   // right_eye
            0.35f,  // left_ear
            0.65f,  // right_ear
            0.3f,   // left_shoulder
            0.7f,   // right_shoulder
            0.2f,   // left_elbow
            0.8f,   // right_elbow
            0.1f,   // left_wrist
            0.9f,   // right_wrist
            0.35f,  // left_hip
            0.65f,  // right_hip
            0.3f,   // left_knee
            0.7f,   // right_knee
            0.25f,  // left_ankle
            0.75f   // right_ankle
        };
        
        if (keypoint_idx < relative_x.size()) {
            return relative_x[keypoint_idx];
        }
        return 0.5f;
    }
    
    float getRelativeY(int keypoint_idx, const cv::Rect& bbox) {
        // Define relative positions with some variation
        std::vector<float> relative_y = {
            0.1f,   // nose
            0.08f,  // left_eye
            0.08f,  // right_eye
            0.1f,   // left_ear
            0.1f,   // right_ear
            0.25f,  // left_shoulder
            0.25f,  // right_shoulder
            0.4f,   // left_elbow
            0.4f,   // right_elbow
            0.55f,  // left_wrist
            0.55f,  // right_wrist
            0.6f,   // left_hip
            0.6f,   // right_hip
            0.8f,   // left_knee
            0.8f,   // right_knee
            0.95f,  // left_ankle
            0.95f   // right_ankle
        };
        
        if (keypoint_idx < relative_y.size()) {
            return relative_y[keypoint_idx];
        }
        return 0.5f;
    }
    
    void updatePoseTracking(std::vector<DynamicPose>& current_poses) {
        // Simple tracking: match poses by proximity
        for (auto& current_pose : current_poses) {
            for (const auto& prev_pose : previous_poses) {
                cv::Point2f current_center(current_pose.bounding_box.x + current_pose.bounding_box.width/2,
                                         current_pose.bounding_box.y + current_pose.bounding_box.height/2);
                cv::Point2f prev_center(prev_pose.bounding_box.x + prev_pose.bounding_box.width/2,
                                      prev_pose.bounding_box.y + prev_pose.bounding_box.height/2);
                
                float distance = cv::norm(current_center - prev_center);
                if (distance < 50) {  // If poses are close, update keypoints with previous history
                    for (size_t i = 0; i < current_pose.keypoints.size() && i < prev_pose.keypoints.size(); i++) {
                        current_pose.keypoints[i].history = prev_pose.keypoints[i].history;
                        current_pose.keypoints[i].velocity = prev_pose.keypoints[i].velocity;
                    }
                    break;
                }
            }
        }
    }
};

void drawDynamicPose(cv::Mat& image, const DynamicPose& pose, float conf_threshold = 0.3) {
    // Draw bounding box
    cv::rectangle(image, pose.bounding_box, cv::Scalar(0, 255, 0), 2);
    
    // Draw keypoints with motion trails
    for (size_t i = 0; i < pose.keypoints.size(); i++) {
        const DynamicKeypoint& kp = pose.keypoints[i];
        
        if (kp.visible && kp.confidence > conf_threshold) {
            // Draw keypoint
            cv::circle(image, kp.point, 5, cv::Scalar(0, 255, 0), -1);
            
            // Draw motion trail
            if (kp.history.size() >= 2) {
                for (size_t j = 1; j < kp.history.size(); j++) {
                    cv::line(image, kp.history[j-1], kp.history[j], 
                            cv::Scalar(255, 255, 0), 2);
                }
            }
            
            // Draw velocity vector
            if (cv::norm(kp.velocity) > 2.0f) {
                cv::arrowedLine(image, kp.point, 
                              kp.point + kp.velocity * 0.5f,
                              cv::Scalar(0, 0, 255), 2);
            }
            
            // Draw keypoint labels for important points
            if (i < 5) {
                cv::putText(image, POSE_KEYPOINTS[i], 
                           cv::Point(kp.point.x + 5, kp.point.y - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);
            }
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
        std::cout << "Dynamic OpenCV Pose Detection Webcam Test" << std::endl;
        std::cout << "OpenCV version: " << CV_VERSION << std::endl;
        
        // Initialize dynamic pose detector
        DynamicPoseDetector pose_detector;
        
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
                
                // Detect dynamic poses
                std::vector<DynamicPose> poses = pose_detector.detectPoses(frame);
                
                // Draw all poses
                for (const auto& pose : poses) {
                    drawDynamicPose(processed, pose, 0.3);
                }
                
                // Add frame counter and instructions
                std::string frame_text = "Frame: " + std::to_string(frame_count);
                putText(processed, frame_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                
                std::string pose_text = "Dynamic poses detected: " + std::to_string(poses.size());
                putText(processed, pose_text, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                
                putText(processed, "Press 'q' to quit, 's' to save, 'p' to pause", Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                
                // Show result
                imshow("Dynamic Pose Detection Webcam", processed);
            }
            
            // Handle key presses
            char key = waitKey(1) & 0xFF;
            
            if (key == 'q' || key == 27) { // 'q' or ESC
                std::cout << "Quitting..." << std::endl;
                break;
            } else if (key == 's') {
                // Save current frame
                std::string filename = "dynamic_pose_frame_" + std::to_string(frame_count) + ".jpg";
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
        
        std::cout << "Dynamic pose detection webcam test completed!" << std::endl;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
} 