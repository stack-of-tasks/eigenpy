cmake_minimum_required(VERSION 2.8.3)

message(STATUS "Loading eigenpy from PkgConfig")

find_package(PkgConfig REQUIRED)
pkg_check_modules(eigenpy REQUIRED eigenpy)

# find absolute library paths for all eigenpy_LIBRARIES
set(libs ${eigenpy_LIBRARIES})
set(eigenpy_LIBRARIES "")
foreach(lib ${libs})
  find_library(abs_lib_${lib} ${lib} HINTS ${eigenpy_LIBRARY_DIRS})
  list(APPEND eigenpy_LIBRARIES "${abs_lib_${lib}}")
endforeach()
