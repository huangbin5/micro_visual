# 设置cmake及项目信息
cmake_minimum_required(VERSION 3.17)
project(patternRecognition)

add_executable(patternRecognition main.cpp)

# 设置OpenCV的路径
find_package(OpenCV REQUIRED)
# 链接OpenCV库
target_link_libraries(patternRecognition ${OpenCV_LIBS})
