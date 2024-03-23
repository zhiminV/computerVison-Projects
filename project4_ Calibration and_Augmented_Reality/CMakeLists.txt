cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
project(Project4)

find_package(OpenCV REQUIRED)
include_directories(/usr/local/include)

#task 1-3 
# add_executable(main main.cpp calibrate.cpp calibrate.h)

# task 4 and task 5
# add_executable(main task4AndTask5.cpp calibrate.cpp calibrate.h)

# task 6
# add_executable(main Draw3Dobject.cpp calibrate.cpp calibrate.h)

# task 7
add_executable(main DetectRobustFeatures.cpp calibrate.cpp calibrate.h)

target_link_libraries(main ${OpenCV_LIBS})
