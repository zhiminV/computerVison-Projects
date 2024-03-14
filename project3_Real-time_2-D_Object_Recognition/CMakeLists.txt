cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
project(Project2)

find_package(OpenCV REQUIRED)
include_directories(/usr/local/include)

add_executable(main main.cpp recognition.cpp recognition.h)
target_link_libraries(main ${OpenCV_LIBS})
