cmake_minimum_required(VERSION 2.8.12)

project(pyncnn)

find_package(ncnn CONFIG REQUIRED)
find_package(pybind11 REQUIRED)

#add_subdirectory(src)

include_directories(${pybind11_INCLUDE_DIR})
pybind11_add_module(pyncnn src/main.cpp)
target_link_libraries(pyncnn PUBLIC ncnn)