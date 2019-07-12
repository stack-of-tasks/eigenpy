#
# Copyright (c) 2014-2019 CNRS
# Copyright (c) 2018-2019 INRIA
#

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)
INCLUDE(cmake/python.cmake)
INCLUDE(cmake/ide.cmake)
INCLUDE(cmake/test.cmake)

SET(PROJECT_NAME eigenpy)
SET(PROJECT_DESCRIPTION "Wrapping Eigen3 -- numpy")
SET(PROJECT_URL "http://github.com/stack-of-tasks/eigenpy")
OPTION(INSTALL_DOCUMENTATION "Generate and install the documentation" FALSE)

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE True)

SETUP_PROJECT()
STRING(REPLACE "-pedantic" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

IF(APPLE)
  SET(CMAKE_MACOSX_RPATH TRUE)
  SET(CMAKE_SKIP_BUILD_RPATH FALSE)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

  #SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  IF("${isSystemDir}" STREQUAL "-1")
     SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  ENDIF("${isSystemDir}" STREQUAL "-1")
ENDIF(APPLE)

IF(WIN32)
  SET(LINK copy_if_different)
ELSE(WIN32)
  SET(LINK create_symlink)
ENDIF(WIN32)


# ----------------------------------------------------
# --- OPTIONS  ---------------------------------------
# ----------------------------------------------------
OPTION (BUILD_UNIT_TESTS "Build the unitary tests" ON)

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
)

INCLUDE_DIRECTORIES(${${PROJECT_NAME}_BINARY_DIR}/include)
INCLUDE_DIRECTORIES(${${PROJECT_NAME}_SOURCE_DIR}/include)

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
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "${PROJECT_NAME}${PYTHON_SOABI}")

TARGET_LINK_BOOST_PYTHON(${PROJECT_NAME})
PKG_CONFIG_USE_DEPENDENCY(${PROJECT_NAME} eigen3)
INSTALL(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

ADD_HEADER_GROUP(${PROJECT_NAME}_HEADERS)
ADD_SOURCE_GROUP(${PROJECT_NAME}_SOURCES)

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

SETUP_PROJECT_FINALIZE()
