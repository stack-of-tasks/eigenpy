cmake_minimum_required(VERSION 3.22)

project(my_project CXX)
find_package(eigenpy 1.0.0 CONFIG REQUIRED)

add_executable(extra_lib extra_lib.cpp)
target_link_libraries(extra_lib PRIVATE eigenpy::eigenpy)
