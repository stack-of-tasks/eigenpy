#
# Copyright (c) 2015-2018 LAAS-CNRS
#
# This file is part of eigenpy.
# eigenpy is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
# eigenpy is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.  You should
# have received a copy of the GNU Lesser General Public License along
# with eigenpy.  If not, see <http://www.gnu.org/licenses/>.
#

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)
INCLUDE(cmake/python.cmake)
INCLUDE(cmake/ide.cmake)

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
  eigenpy/solvers/solvers.hpp
  eigenpy/solvers/preconditioners.hpp
  eigenpy/solvers/IterativeSolverBase.hpp
  eigenpy/solvers/LeastSquaresConjugateGradient.hpp
  eigenpy/solvers/ConjugateGradient.hpp
  eigenpy/solvers/SparseSolverBase.hpp
  eigenpy/solvers/BasicPreconditioners.hpp
  eigenpy/solvers/BFGSPreconditioners.hpp
  )

SET(${PROJECT_NAME}_HEADERS
  ${${PROJECT_NAME}_SOLVERS_HEADERS}
  eigenpy/eigenpy.hpp
  eigenpy/exception.hpp
  eigenpy/details.hpp
  eigenpy/fwd.hpp
  eigenpy/map.hpp
  eigenpy/geometry.hpp
  eigenpy/geometry-conversion.hpp
  eigenpy/memory.hpp
  eigenpy/registration.hpp
  eigenpy/angle-axis.hpp
  eigenpy/quaternion.hpp
  eigenpy/stride.hpp
  eigenpy/ref.hpp
)

EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E ${LINK}
  ${${PROJECT_NAME}_SOURCE_DIR}/src
  ${${PROJECT_NAME}_BINARY_DIR}/eigenpy)
INCLUDE_DIRECTORIES(${${PROJECT_NAME}_BINARY_DIR}/include)
INCLUDE_DIRECTORIES(${${PROJECT_NAME}_BINARY_DIR})

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
PKG_CONFIG_APPEND_CFLAGS("-isystem${PYTHON_INCLUDE_DIRS}")
PKG_CONFIG_APPEND_CFLAGS("-isystem${NUMPY_INCLUDE_DIRS}")
PKG_CONFIG_APPEND_BOOST_LIBS(${BOOST_COMPONENTS})

SETUP_PROJECT_FINALIZE()
