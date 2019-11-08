cmake_minimum_required(VERSION 2.8.3)

message(STATUS "Loading eigenpy from PkgConfig")

find_package(PkgConfig)
pkg_check_modules(eigenpy REQUIRED eigenpy)
link_directories(${eigenpy_LIBRARY_DIRS})
