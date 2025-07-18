#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace dnn;

// COCO class names
std::vector<std::string> class_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

std::vector<Detection> post_process(const cv::Mat& output, const cv::Size& original_size, float conf_threshold = 0.5, float nms_threshold = 0.4) {
    std::vector<Detection> detections;
    
    // YOLOv5 output format: [batch, 25200, 85] where 85 = 4 (bbox) + 1 (conf) + 80 (classes)
    int rows = output.rows;
    int cols = output.cols;
    
    // Process each detection
    for (int i = 0; i < rows; i++) {
        float* row = (float*)output.row(i).data;
        
        // Get confidence scores for all classes
        float* classes_scores = row + 5;
        cv::Mat scores(1, cols - 5, CV_32F, classes_scores);
        cv::Point class_id;
        double max_class_score;
        minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
        
        // Check if confidence is above threshold
        if (max_class_score > conf_threshold) {
            Detection det;
            det.confidence = max_class_score;
            det.class_id = class_id.x;
            
            // Get bounding box coordinates (normalized)
            float x = row[0];
            float y = row[1];
            float w = row[2];
            float h = row[3];
            
            // Convert to pixel coordinates
            int left = int((x - w/2) * original_size.width);
            int top = int((y - h/2) * original_size.height);
            int width = int(w * original_size.width);
            int height = int(h * original_size.height);
            
            det.box = cv::Rect(left, top, width, height);
            detections.push_back(det);
        }
    }
    
    return detections;
}

int main() {
    try {
        // Load model
        std::cout << "Loading YOLOv5 model..." << std::endl;
        Net net = readNetFromONNX("yolov5.onnx");
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
        Size input_size(640, 640);
        
        // Preprocess image
        Mat blob = blobFromImage(img, 1/255.0, input_size, Scalar(), true, false);
        
        // Run inference
        std::cout << "Running inference..." << std::endl;
        net.setInput(blob);
        Mat output = net.forward();
        
        // Post-process detections
        std::cout << "Post-processing detections..." << std::endl;
        std::vector<Detection> detections = post_process(output, original_img.size());
        
        // Draw detections
        std::cout << "Found " << detections.size() << " detections" << std::endl;
        for (const auto& det : detections) {
            std::string label = class_names[det.class_id] + " " + std::to_string(int(det.confidence * 100)) + "%";
            
            // Draw bounding box
            rectangle(original_img, det.box, Scalar(0, 255, 0), 2);
            
            // Draw label
            int baseline = 0;
            Size label_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            rectangle(original_img, 
                     Point(det.box.x, det.box.y - label_size.height - 10),
                     Point(det.box.x + label_size.width, det.box.y),
                     Scalar(0, 255, 0), -1);
            putText(original_img, label, 
                   Point(det.box.x, det.box.y - 5),
                   FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
        }
        
        // Show result
        imshow("YOLOv5 Detections", original_img);
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
