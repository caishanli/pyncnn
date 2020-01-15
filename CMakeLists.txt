cmake_minimum_required(VERSION 2.8.12)

project(pyncnn)

find_package(ncnn REQUIRED)
find_package(pybind11 REQUIRED)

add_subdirectory(src)