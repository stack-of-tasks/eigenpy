cmake_minimum_required(VERSION 2.6)

project(ExtraLib CXX)
find_package(eigenpy REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(SYSTEM ${EIGENPY_INCLUDE_DIRS})
add_executable(extra_lib extra_lib.cpp)
target_link_libraries(extra_lib PUBLIC ${eigenpy_LIBRARIES} ${PYTHON_LIBRARIES})
