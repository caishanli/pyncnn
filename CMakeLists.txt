cmake_minimum_required(VERSION 2.8.12)

project(pyncnn)

set( PACKAGE_VERSION "0.0.1" )

set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

find_package (Python3 COMPONENTS Interpreter Development)
find_package(ncnn CONFIG REQUIRED)
find_package(pybind11 REQUIRED)

if(MSVC OR CMAKE_GENERATOR STREQUAL "Xcode")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${pyncnn_BINARY_DIR}/ncnn/)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${pyncnn_BINARY_DIR}/ncnn/)
endif(MSVC OR CMAKE_GENERATOR STREQUAL "Xcode")

include_directories(${pybind11_INCLUDE_DIR})
pybind11_add_module(pyncnn src/main.cpp)
set_target_properties(pyncnn PROPERTIES OUTPUT_NAME "ncnn")
target_link_libraries(pyncnn PUBLIC ncnn)
set_target_properties(pyncnn PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/ncnn")

configure_file(python/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/ncnn/__init__.py)
configure_file(python/setup.py ${CMAKE_CURRENT_BINARY_DIR}/ncnn/setup.py)