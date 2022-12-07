cmake_minimum_required(VERSION 3.17)
project(test_tnn)

set(CMAKE_CXX_STANDARD 17)


find_package(OpenCV REQUIRED core imgproc highgui)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tnn-master-2022-11-25/include)

add_library(tnn SHARED IMPORTED)
set_target_properties(tnn PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/tnn-master-2022-11-25/lib/libTNN.dylib)


#add_executable(test_tnn main.cpp)
add_executable(test_tnn ./yolov6/yolov6.cpp)
target_link_libraries(test_tnn tnn)
target_link_libraries(test_tnn ${OpenCV_LIBS})