cmake_minimum_required(VERSION 2.6)

project(ExtraLib CXX)
find_package(eigenpy REQUIRED)

add_executable(extra_lib extra_lib.cpp)
target_link_libraries(extra_lib PUBLIC eigenpy::eigenpy)
