cmake_minimum_required(VERSION 3.12)
project(Project1)

find_package(OpenCV REQUIRED)
include_directories(/usr/local/include)

add_executable(main.cpp vidDisplay.cpp filter.cpp filter.h faceDetect.cpp faceDetect.h)
# add_executable(timeBlur timeBlur.cpp filter.cpp)
# add_executable(imgDisplay imgDisplay.cpp)
target_link_libraries(main ${OpenCV_LIBS})
