cmake_minimum_required(VERSION 2.8)

project(demo_facedetection_opencvdnn)

# OpenCV
# If you compile OpenCV yourself, cmake reports an error that it cannot find the OpenCV library. You can manually specify it to the opencv-build folder.
# SET(ENV{OpenCV_DIR} /home/khadas/opencv/opencv-build)

find_package(OpenCV 4.7.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(detect-image detect-image.cpp)
target_link_libraries(detect-image ${OpenCV_LIBS})

