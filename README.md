# OpenCV Deep Learning Examples

This repository contains various OpenCV C++ examples demonstrating different computer vision and deep learning capabilities.

## Prerequisites

- OpenCV 4.5.4 (already installed on your system)
- C++ compiler (g++)
- Webcam (for webcam examples)

## Available Programs

### 1. Basic Image Processing (`basic_test`)
- **File**: `basic_test.cpp`
- **Compile**: `make basic_test`
- **Run**: `./basic_test`
- **Description**: Demonstrates basic OpenCV image processing including grayscale conversion, blurring, and edge detection on a static image.

### 2. Webcam Processing (`webcam_test`)
- **File**: `webcam_test.cpp`
- **Compile**: `make webcam_test`
- **Run**: `./webcam_test`
- **Description**: Real-time webcam processing with grayscale conversion, blurring, and edge detection.
- **Controls**: 
  - `q` or `ESC`: Quit
  - `s`: Save current frame
  - `p`: Pause/resume

### 3. Face Detection Webcam (`face_detection_webcam`)
- **File**: `face_detection_webcam.cpp`
- **Compile**: `make face_detection_webcam`
- **Run**: `./face_detection_webcam`
- **Description**: Real-time face detection using OpenCV's Haar cascade classifier.
- **Controls**: Same as webcam_test

### 4. Pose Detection Webcam (`pose_detection_webcam`)
- **File**: `pose_detection_webcam.cpp`
- **Compile**: `make pose_detection_webcam`
- **Run**: `./pose_detection_webcam`
- **Description**: Real-time pose estimation using HOG person detection with estimated keypoints.
- **Features**: 17 COCO-format keypoints, pose connections, confidence scores
- **Controls**: Same as webcam_test

### 5. Advanced Pose Detection (`advanced_pose_webcam`)
- **File**: `advanced_pose_webcam.cpp`
- **Compile**: `make advanced_pose_webcam`
- **Run**: `./advanced_pose_webcam`
- **Description**: Advanced pose estimation with 33 MediaPipe-format keypoints and DNN model support.
- **Features**: Attempts to download real pose models, falls back to HOG detection
- **Controls**: Same as webcam_test

### 6. Dynamic Pose Detection (`dynamic_pose_webcam`)
- **File**: `dynamic_pose_webcam.cpp`
- **Compile**: `make dynamic_pose_webcam`
- **Run**: `./dynamic_pose_webcam`
- **Description**: **REAL DYNAMIC** pose estimation using motion tracking and background subtraction.
- **Features**: 
  - Motion-based keypoint positioning
  - Motion trails and velocity vectors
  - Background subtraction for motion detection
  - Optical flow tracking
  - Dynamic keypoint history
- **Controls**: Same as webcam_test

### 7. Real OpenPose Detection (`real_pose_webcam`) ⭐
- **File**: `real_pose_webcam.cpp`
- **Compile**: `make real_pose_webcam`
- **Run**: `./real_pose_webcam`
- **Description**: **REAL POSE ESTIMATION** using official OpenPose COCO model with OpenCV DNN.
- **Features**: 
  - 18 COCO keypoints (nose, neck, shoulders, elbows, wrists, hips, knees, ankles, eyes, ears)
  - Real-time pose detection from webcam
  - Accurate keypoint positioning
  - Proper skeleton connections
  - Uses official OpenPose model (200MB)
- **Requirements**: `pose_coco.caffemodel` and `pose_coco.prototxt`
- **Controls**: Same as webcam_test

### 8. YOLOv5 Object Detection (`test`)
- **File**: `test.cpp`
- **Compile**: `make test`
- **Run**: `./test`
- **Description**: YOLOv5 object detection on static images (requires compatible ONNX model).
- **Note**: May have compatibility issues with OpenCV 4.5.4 due to FLOAT16 data type.

### 9. MobileNet-SSD Object Detection (`simple_test`)
- **File**: `simple_test.cpp`
- **Compile**: `make simple_test`
- **Run**: `./simple_test`
- **Description**: MobileNet-SSD object detection (requires model download).
- **Note**: Model download may fail due to Google Drive restrictions.

## Compilation

### Compile All Programs
```bash
make all
```

### Compile Individual Programs
```bash
make basic_test          # Basic image processing
make webcam_test         # Webcam processing
make face_detection_webcam # Face detection
make pose_detection_webcam # Pose detection
make advanced_pose_webcam # Advanced pose detection
make test               # YOLOv5 detection
make simple_test        # MobileNet-SSD detection
```

### Clean Build Files
```bash
make clean
```

## Webcam Usage

The webcam programs are configured to use device 1 by default. If device 1 is not available, they will automatically fall back to device 0.

## Troubleshooting

### YOLOv5 Issues
- The YOLOv5 ONNX model may not work with OpenCV 4.5.4 due to FLOAT16 data type support
- Consider using the face detection or basic webcam examples instead

### Webcam Issues
- Ensure your webcam is properly connected and accessible
- Try different device numbers (0, 1, 2, etc.) if needed
- Check webcam permissions

### Model Download Issues
- The MobileNet-SSD model download may fail due to Google Drive restrictions
- Use the face detection example which uses built-in OpenCV models

## File Structure

- `*.cpp`: Source code files
- `*.hpp`: Header files (if any)
- `Makefile`: Build configuration
- `image.jpg`: Test image for static processing
- `yolov5.onnx`: YOLOv5 model file (14MB)
- `MobileNetSSD_deploy.prototxt`: MobileNet-SSD configuration file

## Key Features Demonstrated

1. **Image Processing**: Grayscale conversion, blurring, edge detection
2. **Video Capture**: Real-time webcam processing
3. **Object Detection**: Face detection using Haar cascades
4. **Pose Estimation**: Real-time pose detection with keypoint tracking
5. **Deep Learning**: DNN module usage with pre-trained models
5. **User Interface**: Real-time display with keyboard controls
6. **Error Handling**: Robust error checking and fallback mechanisms 