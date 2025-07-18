#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <fstream>

using namespace cv;
using namespace dnn;

bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

int main() {
    try {
        // Load a pre-trained model that comes with OpenCV
        std::cout << "Loading MobileNet-SSD model..." << std::endl;
        
        // Download model files if they don't exist
        std::string model_path = "MobileNetSSD_deploy.caffemodel";
        std::string config_path = "MobileNetSSD_deploy.prototxt";
        
        if (!fileExists(model_path) || !fileExists(config_path)) {
            std::cout << "Downloading MobileNet-SSD model files..." << std::endl;
            system("wget https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt -O MobileNetSSD_deploy.prototxt");
            system("wget https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc -O MobileNetSSD_deploy.caffemodel");
        }
        
        Net net = readNetFromCaffe(config_path, model_path);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);

        // Load image
        std::cout << "Loading image..." << std::endl;
        Mat img = imread("image.jpg");
        if (img.empty()) {
            std::cerr << "Error: Could not load image.jpg" << std::endl;
            return -1;
        }
        
        Mat original_img = img.clone();
        Size input_size(300, 300);
        
        // Preprocess image
        Mat blob = blobFromImage(img, 0.007843, input_size, Scalar(127.5, 127.5, 127.5), false, false);
        
        // Run inference
        std::cout << "Running inference..." << std::endl;
        net.setInput(blob);
        Mat output = net.forward();
        
        // MobileNet-SSD class names
        std::vector<std::string> class_names = {
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        };
        
        // Process detections
        std::cout << "Processing detections..." << std::endl;
        Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());
        
        float confidenceThreshold = 0.5;
        int num_detections = 0;
        
        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);
            
            if (confidence > confidenceThreshold) {
                int class_id = static_cast<int>(detectionMat.at<float>(i, 1));
                int left = static_cast<int>(detectionMat.at<float>(i, 3) * original_img.cols);
                int top = static_cast<int>(detectionMat.at<float>(i, 4) * original_img.rows);
                int right = static_cast<int>(detectionMat.at<float>(i, 5) * original_img.cols);
                int bottom = static_cast<int>(detectionMat.at<float>(i, 6) * original_img.rows);
                
                // Ensure coordinates are within image bounds
                left = std::max(0, std::min(left, original_img.cols - 1));
                top = std::max(0, std::min(top, original_img.rows - 1));
                right = std::max(left + 1, std::min(right, original_img.cols));
                bottom = std::max(top + 1, std::min(bottom, original_img.rows));
                
                // Draw bounding box
                rectangle(original_img, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);
                
                // Draw label
                std::string label = class_names[class_id] + " " + std::to_string(int(confidence * 100)) + "%";
                int baseline = 0;
                Size label_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                rectangle(original_img, 
                         Point(left, top - label_size.height - 10),
                         Point(left + label_size.width, top),
                         Scalar(0, 255, 0), -1);
                putText(original_img, label, 
                       Point(left, top - 5),
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
                
                num_detections++;
            }
        }
        
        std::cout << "Found " << num_detections << " detections" << std::endl;
        
        // Show result
        imshow("MobileNet-SSD Detections", original_img);
        std::cout << "Press any key to exit..." << std::endl;
        waitKey(0);
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
} 