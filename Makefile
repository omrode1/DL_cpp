CXX = g++
CXXFLAGS = -std=c++11 $(shell pkg-config --cflags opencv4)
LDFLAGS = $(shell pkg-config --libs opencv4)

TARGET = test
SIMPLE_TARGET = simple_test
BASIC_TARGET = basic_test
WEBCAM_TARGET = webcam_test
FACE_TARGET = face_detection_webcam
SOURCE = test.cpp
SIMPLE_SOURCE = simple_test.cpp
BASIC_SOURCE = basic_test.cpp
WEBCAM_SOURCE = webcam_test.cpp
FACE_SOURCE = face_detection_webcam.cpp

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)

$(SIMPLE_TARGET): $(SIMPLE_SOURCE)
	$(CXX) $(CXXFLAGS) -o $(SIMPLE_TARGET) $(SIMPLE_SOURCE) $(LDFLAGS)

$(BASIC_TARGET): $(BASIC_SOURCE)
	$(CXX) $(CXXFLAGS) -o $(BASIC_TARGET) $(BASIC_SOURCE) $(LDFLAGS)

$(WEBCAM_TARGET): $(WEBCAM_SOURCE)
	$(CXX) $(CXXFLAGS) -o $(WEBCAM_TARGET) $(WEBCAM_SOURCE) $(LDFLAGS)

$(FACE_TARGET): $(FACE_SOURCE)
	$(CXX) $(CXXFLAGS) -o $(FACE_TARGET) $(FACE_SOURCE) $(LDFLAGS)

all: $(TARGET) $(SIMPLE_TARGET) $(BASIC_TARGET) $(WEBCAM_TARGET) $(FACE_TARGET)

clean:
	rm -f $(TARGET) $(SIMPLE_TARGET) $(BASIC_TARGET) $(WEBCAM_TARGET) $(FACE_TARGET)

.PHONY: clean all 