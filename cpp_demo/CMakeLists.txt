cmake_minimum_required(VERSION 2.8)

project(demo_facedetection_opencvdnn)

# OpenCV
# 如果是自己编译的OpenCV，cmake报错找不到OpenCV库，可以手动指定到opencv-build文件夹
# SET(ENV{OpenCV_DIR} /home/khadas/opencv/opencv-build)

find_package(OpenCV 4.7.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(detect-image detect-image.cpp priorbox.cpp utils.cpp)
target_link_libraries(detect-image ${OpenCV_LIBS})

add_executable(detect-camera detect-camera.cpp priorbox.cpp utils.cpp)
target_link_libraries(detect-camera ${OpenCV_LIBS})
