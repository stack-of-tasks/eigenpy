#
# Copyright (c) 2014-2019 CNRS Copyright (c) 2018-2023 INRIA
#

cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME eigenpy)
set(PROJECT_DESCRIPTION "Bindings between Numpy and Eigen using Boost.Python")
set(PROJECT_URL "http://github.com/stack-of-tasks/eigenpy")
set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(PROJECT_USE_KEYWORD_LINK_LIBRARIES TRUE)
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_COMPATIBILITY_VERSION AnyNewerVersion)

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  if(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    message(
      FATAL_ERROR
        "\nPlease run the following command first:\ngit submodule update --init\n"
    )
  else()
    message(STATUS "JRL cmakemodules not found. Let's fetch it.")
    include(FetchContent)
    FetchContent_Declare(
      "jrl-cmakemodules"
      GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  endif()
endif()

# Disable -Werror on Unix for now.
set(CXX_DISABLE_WERROR True)
set(CMAKE_VERBOSE_MAKEFILE True)

# ----------------------------------------------------
# --- OPTIONS  ---------------------------------------
# Need to be set before including base.cmake
# ----------------------------------------------------
option(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)
option(SUFFIX_SO_VERSION "Suffix library name with its version" OFF)

if(DEFINED BUILD_UNIT_TESTS)
  message(
    AUTHOR_WARNING
      "BUILD_UNIT_TESTS is deprecated. Use BUILD_TESTING instead.\
    If you are manually building EigenPy from source in an existing build folder,\
    we suggest that you delete your build folder and make a new one.")
  set(BUILD_TESTING ${BUILD_UNIT_TESTS})
endif(DEFINED BUILD_UNIT_TESTS)

include("${JRL_CMAKE_MODULES}/base.cmake")
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

include("${JRL_CMAKE_MODULES}/boost.cmake")
include("${JRL_CMAKE_MODULES}/python.cmake")
include("${JRL_CMAKE_MODULES}/ide.cmake")
include("${JRL_CMAKE_MODULES}/apple.cmake")

option(GENERATE_PYTHON_STUBS
       "Generate the Python stubs associated to the Python library" OFF)

string(REPLACE "-pedantic" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

# If needed, fix CMake policy for APPLE systems
apply_default_apple_configuration()
check_minimal_cxx_standard(11 ENFORCE)

if(WIN32)
  set(LINK copy_if_different)
else(WIN32)
  set(LINK create_symlink)
endif(WIN32)

if(CMAKE_CROSSCOMPILING)
  set(PYTHON_COMPONENTS Interpreter NumPy)
else()
  set(PYTHON_COMPONENTS Interpreter Development.Module NumPy)
endif()
set(PYTHON_EXPORT_DEPENDENCY ON)
findpython(REQUIRED)

if(${NUMPY_VERSION} VERSION_LESS "1.16.0")
  set(NUMPY_WITH_BROKEN_UFUNC_SUPPORT TRUE)
endif()

if(WIN32)
  link_directories(${PYTHON_LIBRARY_DIRS})
  # # Set default Windows build paths SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY
  # ${PROJECT_BINARY_DIR}/Bin CACHE PATH "Single directory for all libraries")
  # SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/Bin CACHE PATH
  # "Single directory for all executables") SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  # ${PROJECT_BINARY_DIR}/Bin CACHE PATH "Sing$le directory for all archives")
endif(WIN32)

# ----------------------------------------------------
# --- DEPENDENCIES -----------------------------------
# ----------------------------------------------------
add_project_dependency(Eigen3 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.0.5")

set_boost_default_options()
export_boost_default_options()
find_package(Boost REQUIRED)
search_for_boost_python(REQUIRED)

# ----------------------------------------------------
# --- INCLUDE ----------------------------------------
# ----------------------------------------------------
set(${PROJECT_NAME}_UTILS_HEADERS
    include/eigenpy/utils/scalar-name.hpp include/eigenpy/utils/is-approx.hpp
    include/eigenpy/utils/is-aligned.hpp)

set(${PROJECT_NAME}_SOLVERS_HEADERS
    include/eigenpy/solvers/solvers.hpp
    include/eigenpy/solvers/preconditioners.hpp
    include/eigenpy/solvers/IterativeSolverBase.hpp
    include/eigenpy/solvers/LeastSquaresConjugateGradient.hpp
    include/eigenpy/solvers/ConjugateGradient.hpp
    include/eigenpy/solvers/SparseSolverBase.hpp
    include/eigenpy/solvers/BasicPreconditioners.hpp
    include/eigenpy/solvers/BFGSPreconditioners.hpp)

set(${PROJECT_NAME}_DECOMPOSITIONS_HEADERS
    include/eigenpy/decompositions/decompositions.hpp
    include/eigenpy/decompositions/EigenSolver.hpp
    include/eigenpy/decompositions/LDLT.hpp
    include/eigenpy/decompositions/LLT.hpp
    include/eigenpy/decompositions/SelfAdjointEigenSolver.hpp
    include/eigenpy/decompositions/minres.hpp)

set(${PROJECT_NAME}_HEADERS
    ${${PROJECT_NAME}_UTILS_HEADERS}
    ${${PROJECT_NAME}_SOLVERS_HEADERS}
    ${${PROJECT_NAME}_DECOMPOSITIONS_HEADERS}
    include/eigenpy/alignment.hpp
    include/eigenpy/computation-info.hpp
    include/eigenpy/eigenpy.hpp
    include/eigenpy/exception.hpp
    include/eigenpy/scalar-conversion.hpp
    include/eigenpy/expose.hpp
    include/eigenpy/copyable.hpp
    include/eigenpy/details.hpp
    include/eigenpy/fwd.hpp
    include/eigenpy/eigen-allocator.hpp
    include/eigenpy/eigen-to-python.hpp
    include/eigenpy/eigen-from-python.hpp
    include/eigenpy/eigen-typedef.hpp
    include/eigenpy/numpy-map.hpp
    include/eigenpy/geometry.hpp
    include/eigenpy/geometry-conversion.hpp
    include/eigenpy/memory.hpp
    include/eigenpy/numpy.hpp
    include/eigenpy/numpy-allocator.hpp
    include/eigenpy/numpy-type.hpp
    include/eigenpy/registration.hpp
    include/eigenpy/registration_class.hpp
    include/eigenpy/angle-axis.hpp
    include/eigenpy/quaternion.hpp
    include/eigenpy/user-type.hpp
    include/eigenpy/ufunc.hpp
    include/eigenpy/register.hpp
    include/eigenpy/std-map.hpp
    include/eigenpy/std-vector.hpp
    include/eigenpy/optional.hpp
    include/eigenpy/pickle-vector.hpp
    include/eigenpy/stride.hpp
    include/eigenpy/tensor/eigen-from-python.hpp
    include/eigenpy/swig.hpp
    include/eigenpy/version.hpp)

list(
  APPEND
  ${PROJECT_NAME}_HEADERS
  ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/config.hpp
  ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/deprecated.hpp
  ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/warning.hpp)

# ----------------------------------------------------
# --- TARGETS ----------------------------------------
# ----------------------------------------------------
set(${PROJECT_NAME}_SOLVERS_SOURCES src/solvers/preconditioners.cpp
                                    src/solvers/solvers.cpp)

set(${PROJECT_NAME}_DECOMPOSITIONS_SOURCES
    src/decompositions/decompositions.cpp)

set(${PROJECT_NAME}_SOURCES
    ${${PROJECT_NAME}_SOLVERS_SOURCES}
    ${${PROJECT_NAME}_DECOMPOSITIONS_SOURCES}
    src/exception.cpp
    src/eigenpy.cpp
    src/numpy.cpp
    src/numpy-type.cpp
    src/matrix-float.cpp
    src/matrix-complex-float.cpp
    src/matrix-complex-double.cpp
    src/register.cpp
    src/matrix-double.cpp
    src/matrix-long-double.cpp
    src/matrix-complex-long-double.cpp
    src/matrix-bool.cpp
    src/matrix-int.cpp
    src/matrix-long.cpp
    src/angle-axis.cpp
    src/quaternion.cpp
    src/geometry-conversion.cpp
    src/std-vector.cpp
    src/optional.cpp
    src/version.cpp)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES}
                                   ${${PROJECT_NAME}_HEADERS})
target_include_directories(
  ${PROJECT_NAME} SYSTEM
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:include>)

modernize_target_link_libraries(
  ${PROJECT_NAME}
  SCOPE
  PUBLIC
  TARGETS
  Eigen3::Eigen
  INCLUDE_DIRS
  ${EIGEN3_INCLUDE_DIR})

modernize_target_link_libraries(
  ${PROJECT_NAME}
  SCOPE
  PUBLIC
  TARGETS
  Python${PYTHON_VERSION_MAJOR}::NumPy
  INCLUDE_DIRS
  ${NUMPY_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIR})

if(SUFFIX_SO_VERSION)
  set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION})
endif(SUFFIX_SO_VERSION)

if(NOT WIN32)
  target_compile_options(
    ${PROJECT_NAME} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:-bigobj -MP>
                            "-Wno-conversion")
else()
  target_compile_options(${PROJECT_NAME}
                         PRIVATE $<$<CXX_COMPILER_ID:MSVC>:-bigobj -MP>)
  target_compile_definitions(${PROJECT_NAME} PUBLIC "HAVE_SNPRINTF")
endif()

target_link_boost_python(${PROJECT_NAME} PUBLIC)
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${TARGETS_EXPORT_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_header_group(${PROJECT_NAME}_HEADERS)
add_source_group(${PROJECT_NAME}_SOURCES)

# Install package for ROS
install(FILES package.xml DESTINATION share/eigenpy)
# Allows Colcon to find non-Ament packages when using workspace underlays
file(
  WRITE
  ${CMAKE_CURRENT_BINARY_DIR}/share/ament_index/resource_index/packages/${PROJECT_NAME}
  "")
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/share/ament_index/resource_index/packages/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages)
file(
  WRITE
  ${CMAKE_CURRENT_BINARY_DIR}/share/${PROJECT_NAME}/hook/ament_prefix_path.dsv
  "prepend-non-duplicate;AMENT_PREFIX_PATH;")
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/share/${PROJECT_NAME}/hook/ament_prefix_path.dsv
  DESTINATION share/${PROJECT_NAME}/hook)
file(WRITE
     ${CMAKE_CURRENT_BINARY_DIR}/share/${PROJECT_NAME}/hook/python_path.dsv
     "prepend-non-duplicate;PYTHONPATH;${PYTHON_SITELIB}")
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/share/${PROJECT_NAME}/hook/python_path.dsv
  DESTINATION share/${PROJECT_NAME}/hook)

# ----------------------------------------------------
# --- PYTHON LIBRARY ---------------------------------
# ----------------------------------------------------
add_subdirectory(python)

# ----------------------------------------------------
# --- UNIT TEST --------------------------------------
# ----------------------------------------------------
add_subdirectory(unittest)

pkg_config_append_libs(${PROJECT_NAME})
pkg_config_append_cflags("-I${PYTHON_INCLUDE_DIRS}")
pkg_config_append_cflags("-I${NUMPY_INCLUDE_DIRS}")
pkg_config_append_boost_libs(${BOOST_COMPONENTS})
