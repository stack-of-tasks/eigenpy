#
# Copyright (c) 2014-2019 CNRS
# Copyright (c) 2018-2019 INRIA
#

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(PROJECT_NAME eigenpy)
SET(PROJECT_DESCRIPTION "Bindings between Numpy and Eigen using Boost.Python")
SET(PROJECT_URL "http://github.com/stack-of-tasks/eigenpy")

# Check if the submodule cmake have been initialized
IF(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  MESSAGE(FATAL_ERROR "\nPlease run the following command first:\ngit submodule update --init\n")
ENDIF()

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE True)

INCLUDE(cmake/base.cmake)
IF(CMAKE_VERSION VERSION_GREATER "3.0.0")
  CMAKE_POLICY(SET CMP0048 NEW)
  PROJECT(${PROJECT_NAME} VERSION ${PROJECT_VERSION_FULL} LANGUAGES CXX DESCRIPTION ${PROJECT_DESCRIPTION})
ELSE(CMAKE_VERSION VERSION_GREATER "3.0.0")
  PROJECT(${PROJECT_NAME} CXX)
ENDIF(CMAKE_VERSION VERSION_GREATER "3.0.0")

INCLUDE(cmake/boost.cmake)
INCLUDE(cmake/python.cmake)
INCLUDE(cmake/ide.cmake)
INCLUDE(cmake/apple.cmake)

STRING(REPLACE "-pedantic" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

# If needed, fix CMake policy for APPLE systems
APPLY_DEFAULT_APPLE_CONFIGURATION()

IF(WIN32)
  SET(LINK copy_if_different)
ELSE(WIN32)
  SET(LINK create_symlink)
ENDIF(WIN32)

# ----------------------------------------------------
# --- OPTIONS  ---------------------------------------
# ----------------------------------------------------
OPTION(INSTALL_DOCUMENTATION "Generate and install the documentation" FALSE)
OPTION(BUILD_UNIT_TESTS "Build the unitary tests" OFF)

IF(BUILD_UNIT_TESTS)
  SET(BUILD_TESTING ON)
ELSE(BUILD_UNIT_TESTS)
  SET(BUILD_TESTING OFF) 
ENDIF(BUILD_UNIT_TESTS)

FINDPYTHON()

INCLUDE_DIRECTORIES(SYSTEM ${PYTHON_INCLUDE_DIRS})
FIND_NUMPY()
INCLUDE_DIRECTORIES(SYSTEM ${NUMPY_INCLUDE_DIRS})

# ----------------------------------------------------
# --- DEPENDANCIES -----------------------------------
# ----------------------------------------------------
ADD_REQUIRED_DEPENDENCY("eigen3 >= 3.0.5")

SET(BOOST_COMPONENTS python)
SEARCH_FOR_BOOST()
# Add Boost path to include directories.
INCLUDE_DIRECTORIES(SYSTEM ${Boost_INCLUDE_DIRS})

# ----------------------------------------------------
# --- INCLUDE ----------------------------------------
# ----------------------------------------------------
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

SET(${PROJECT_NAME}_HEADERS
  ${${PROJECT_NAME}_SOLVERS_HEADERS}
  include/eigenpy/eigenpy.hpp
  include/eigenpy/exception.hpp
  include/eigenpy/expose.hpp
  include/eigenpy/details.hpp
  include/eigenpy/fwd.hpp
  include/eigenpy/map.hpp
  include/eigenpy/geometry.hpp
  include/eigenpy/geometry-conversion.hpp
  include/eigenpy/memory.hpp
  include/eigenpy/registration.hpp
  include/eigenpy/angle-axis.hpp
  include/eigenpy/quaternion.hpp
  include/eigenpy/stride.hpp
  include/eigenpy/ref.hpp
  include/eigenpy/details/rvalue_from_python_data.hpp
)

INCLUDE_DIRECTORIES(${${PROJECT_NAME}_BINARY_DIR}/include)
INCLUDE_DIRECTORIES(${${PROJECT_NAME}_SOURCE_DIR}/include)

LINK_DIRECTORIES(${PYTHON_LIBRARY_DIRS})

# ----------------------------------------------------
# --- TARGETS ----------------------------------------
# ----------------------------------------------------
SET(${PROJECT_NAME}_SOLVERS_SOURCES
  src/solvers/preconditioners.cpp
  src/solvers/solvers.cpp
  )

SET(${PROJECT_NAME}_SOURCES
  ${${PROJECT_NAME}_SOLVERS_SOURCES}
  src/exception.cpp
  src/eigenpy.cpp
  src/angle-axis.cpp
  src/quaternion.cpp
  src/geometry-conversion.cpp
)

ADD_LIBRARY(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
SET(LIB_${PROJECT_NAME}_SUFFIX ${PYTHON_EXT_SUFFIX})
IF(APPLE)
  STRING(REPLACE ".so" ".dylib" LIB_${PROJECT_NAME}_SUFFIX ${LIB_${PROJECT_NAME}_SUFFIX})
ENDIF(APPLE)
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES SUFFIX "${LIB_${PROJECT_NAME}_SUFFIX}")
TARGET_COMPILE_OPTIONS(${PROJECT_NAME} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:-bigobj>)

TARGET_LINK_BOOST_PYTHON(${PROJECT_NAME})
PKG_CONFIG_USE_DEPENDENCY(${PROJECT_NAME} eigen3)
INSTALL(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

ADD_HEADER_GROUP(${PROJECT_NAME}_HEADERS)
ADD_SOURCE_GROUP(${PROJECT_NAME}_SOURCES)

INCLUDE(GenerateExportHeader)
GENERATE_EXPORT_HEADER(${PROJECT_NAME} EXPORT_FILE_NAME eigenpy/${PROJECT_NAME}_export.h)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")

INSTALL(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/eigenpy/${PROJECT_NAME}_export.h
  DESTINATION include/eigenpy)
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
