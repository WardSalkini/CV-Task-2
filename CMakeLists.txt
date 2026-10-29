cmake_minimum_required(VERSION 3.15)
project(canny_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

add_library(canny_engine SHARED canny_engine.cpp)
target_include_directories(canny_engine PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(canny_engine PRIVATE ${OpenCV_LIBS})

# canny_engine.dll lands next to canny_app.py
set_target_properties(canny_engine PROPERTIES
        OUTPUT_NAME "canny_engine"
        PREFIX ""
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

if (MSVC)
    target_compile_options(canny_engine PRIVATE /O2)
else()
    target_compile_options(canny_engine PRIVATE -O3)
endif()