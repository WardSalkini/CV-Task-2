cmake_minimum_required(VERSION 3.15)
project(engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

add_library(engine SHARED engine.cpp)
target_include_directories(engine PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(engine PRIVATE ${OpenCV_LIBS})

# engine.dll lands next to app.py
set_target_properties(engine PROPERTIES
        PREFIX ""
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})