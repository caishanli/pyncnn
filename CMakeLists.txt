cmake_minimum_required(VERSION 2.8.12)

project(pyncnn)

set( PACKAGE_VERSION "0.0.1" )

set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

find_package(ncnn CONFIG REQUIRED)
find_package(pybind11 REQUIRED)

#add_subdirectory(src)

include_directories(${pybind11_INCLUDE_DIR})
pybind11_add_module(pyncnn src/main.cpp)
target_link_libraries(pyncnn PUBLIC ncnn)

set_target_properties(pyncnn PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pyncnn")
configure_file(python/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/pyncnn/__init__.py)
configure_file(python/setup.py ${CMAKE_CURRENT_BINARY_DIR}/pyncnn/setup.py)