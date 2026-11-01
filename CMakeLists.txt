cmake_minimum_required(VERSION 3.15)
project(engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# One library for the whole assignment (Canny, Hough, snake, chain code). No OpenCV needed.
add_library(engine SHARED engine.cpp)

# engine.dll lands next to app.py
set_target_properties(engine PROPERTIES
        OUTPUT_NAME "engine"
        PREFIX ""
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})

if (MSVC)
    target_compile_options(engine PRIVATE $<$<NOT:$<CONFIG:Debug>>:/O2>)
else()
    target_compile_options(engine PRIVATE -O3)
endif()
