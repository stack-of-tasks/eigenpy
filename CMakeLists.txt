#
# Copyright (c) 2014-2019 CNRS
# Copyright (c) 2018-2022 INRIA
#

CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

SET(PROJECT_NAME eigenpy)
SET(PROJECT_DESCRIPTION "Bindings between Numpy and Eigen using Boost.Python")
SET(PROJECT_URL "http://github.com/stack-of-tasks/eigenpy")
SET(PROJECT_USE_CMAKE_EXPORT TRUE)
SET(PROJECT_USE_KEYWORD_LINK_LIBRARIES TRUE)
SET(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
SET(PROJECT_COMPATIBILITY_VERSION AnyNewerVersion)

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
IF(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  IF(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    MESSAGE(FATAL_ERROR "\nPlease run the following command first:\ngit submodule update --init\n")
  ELSE()
    message(STATUS "JRL cmakemodules not found. Let's fetch it.")
    include(FetchContent)
    FetchContent_Declare("jrl-cmakemodules"
      GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  ENDIF()
ENDIF()

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE True)

# ----------------------------------------------------
# --- OPTIONS  ---------------------------------------
# Need to be set before including base.cmake
# ----------------------------------------------------
OPTION(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)
OPTION(SUFFIX_SO_VERSION "Suffix library name with its version" OFF)

IF(DEFINED BUILD_UNIT_TESTS)
  MESSAGE(AUTHOR_WARNING "BUILD_UNIT_TESTS is deprecated. Use BUILD_TESTING instead.\
    If you are manually building EigenPy from source in an existing build folder,\
    we suggest that you delete your build folder and make a new one.")
  SET(BUILD_TESTING ${BUILD_UNIT_TESTS})
ENDIF(DEFINED BUILD_UNIT_TESTS)

INCLUDE("${JRL_CMAKE_MODULES}/base.cmake")
COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
PROJECT(${PROJECT_NAME} ${PROJECT_ARGS})

INCLUDE("${JRL_CMAKE_MODULES}/boost.cmake")
INCLUDE("${JRL_CMAKE_MODULES}/python.cmake")
INCLUDE("${JRL_CMAKE_MODULES}/ide.cmake")
INCLUDE("${JRL_CMAKE_MODULES}/apple.cmake")

OPTION(GENERATE_PYTHON_STUBS "Generate the Python stubs associated to the Python library" OFF)

STRING(REPLACE "-pedantic" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

# If needed, fix CMake policy for APPLE systems
APPLY_DEFAULT_APPLE_CONFIGURATION()

IF(WIN32)
  SET(LINK copy_if_different)
ELSE(WIN32)
  SET(LINK create_symlink)
ENDIF(WIN32)

FINDPYTHON()
FIND_NUMPY()

IF(WIN32)
  LINK_DIRECTORIES(${PYTHON_LIBRARY_DIRS})
#  # Set default Windows build paths
#  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY
#    ${PROJECT_BINARY_DIR}/Bin
#    CACHE PATH "Single directory for all libraries")
#  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY
#    ${PROJECT_BINARY_DIR}/Bin
#    CACHE PATH "Single directory for all executables")
#  SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
#    ${PROJECT_BINARY_DIR}/Bin
#    CACHE PATH "Sing$le directory for all archives")
ENDIF(WIN32)

# ----------------------------------------------------
# --- DEPENDENCIES -----------------------------------
# ----------------------------------------------------
ADD_PROJECT_DEPENDENCY(Eigen3 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.0.5")

SET_BOOST_DEFAULT_OPTIONS()
EXPORT_BOOST_DEFAULT_OPTIONS()
FIND_PACKAGE(Boost REQUIRED)
SEARCH_FOR_BOOST_PYTHON()

# ----------------------------------------------------
# --- INCLUDE ----------------------------------------
# ----------------------------------------------------
SET(${PROJECT_NAME}_UTILS_HEADERS
  include/eigenpy/utils/scalar-name.hpp
  include/eigenpy/utils/is-approx.hpp
  include/eigenpy/utils/is-aligned.hpp
  )

SET(${PROJECT_NAME}_SOLVERS_HEADERS
  include/eigenpy/solvers/solvers.hpp
  include/eigenpy/solvers/preconditioners.hpp
  include/eigenpy/solvers/IterativeSolverBase.hpp
  include/eigenpy/solvers/LeastSquaresConjugateGradient.hpp
  include/eigenpy/solvers/ConjugateGradient.hpp
  include/eigenpy/solvers/SparseSolverBase.hpp
  include/eigenpy/solvers/BasicPreconditioners.hpp
  include/eigenpy/solvers/BFGSPreconditioners.hpp
  )

SET(${PROJECT_NAME}_DECOMPOSITIONS_HEADERS
  include/eigenpy/decompositions/decompositions.hpp
  include/eigenpy/decompositions/EigenSolver.hpp
  include/eigenpy/decompositions/LDLT.hpp
  include/eigenpy/decompositions/LLT.hpp
  include/eigenpy/decompositions/SelfAdjointEigenSolver.hpp
  include/eigenpy/decompositions/minres.hpp
  )

SET(${PROJECT_NAME}_HEADERS
  ${${PROJECT_NAME}_UTILS_HEADERS}
  ${${PROJECT_NAME}_SOLVERS_HEADERS}
  ${${PROJECT_NAME}_DECOMPOSITIONS_HEADERS}
  include/eigenpy/computation-info.hpp
  include/eigenpy/eigenpy.hpp
  include/eigenpy/exception.hpp
  include/eigenpy/scalar-conversion.hpp
  include/eigenpy/expose.hpp
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
  include/eigenpy/angle-axis.hpp
  include/eigenpy/quaternion.hpp
  include/eigenpy/user-type.hpp
  include/eigenpy/ufunc.hpp
  include/eigenpy/register.hpp
  include/eigenpy/stride.hpp
  include/eigenpy/swig.hpp
  include/eigenpy/version.hpp
)

LIST(APPEND ${PROJECT_NAME}_HEADERS
  ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/config.hpp
  ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/deprecated.hpp
  ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/warning.hpp)

# ----------------------------------------------------
# --- TARGETS ----------------------------------------
# ----------------------------------------------------
SET(${PROJECT_NAME}_SOLVERS_SOURCES
  src/solvers/preconditioners.cpp
  src/solvers/solvers.cpp
  )

SET(${PROJECT_NAME}_DECOMPOSITIONS_SOURCES
  src/decompositions/decompositions.cpp
  )

SET(${PROJECT_NAME}_SOURCES
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
  src/version.cpp
)

ADD_LIBRARY(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME}
  SYSTEM PUBLIC
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

IF(SUFFIX_SO_VERSION)
  SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION})
ENDIF(SUFFIX_SO_VERSION)

IF(NOT WIN32)
  TARGET_COMPILE_OPTIONS(${PROJECT_NAME} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:-bigobj> "-Wno-conversion")
ELSE()
  TARGET_COMPILE_OPTIONS(${PROJECT_NAME} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:-bigobj>)
ENDIF()

TARGET_LINK_BOOST_PYTHON(${PROJECT_NAME} PUBLIC)
INSTALL(TARGETS ${PROJECT_NAME}
  EXPORT ${TARGETS_EXPORT_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR})

ADD_HEADER_GROUP(${PROJECT_NAME}_HEADERS)
ADD_SOURCE_GROUP(${PROJECT_NAME}_SOURCES)

# Install package for ROS
install(FILES package.xml DESTINATION share/eigenpy)
# Allows Colcon to find non-Ament packages when using workspace underlays
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/share/ament_index/resource_index/packages/${PROJECT_NAME} "")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/share/ament_index/resource_index/packages/${PROJECT_NAME} DESTINATION share/ament_index/resource_index/packages)

# ----------------------------------------------------
# --- PYTHON LIBRARY ---------------------------------
# ----------------------------------------------------
ADD_SUBDIRECTORY(python)

# ----------------------------------------------------
# --- UNIT TEST --------------------------------------
# ----------------------------------------------------
ADD_SUBDIRECTORY(unittest)

PKG_CONFIG_APPEND_LIBS(${PROJECT_NAME})
PKG_CONFIG_APPEND_CFLAGS("-I${PYTHON_INCLUDE_DIRS}")
PKG_CONFIG_APPEND_CFLAGS("-I${NUMPY_INCLUDE_DIRS}")
PKG_CONFIG_APPEND_BOOST_LIBS(${BOOST_COMPONENTS})
