#
#  Copyright 2014 CNRS
#

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)
INCLUDE(cmake/cpack.cmake)
INCLUDE(cmake/eigen.cmake)
INCLUDE(cmake/python.cmake)

SET(PROJECT_NAME eigenpy)
SET(PROJECT_DESCRIPTION "Wrapping Eigen3 -- numpy")
SET(PROJECT_URL "http://github.com/stac-of-tasks/eigenpy")

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE True)

SETUP_PROJECT()
STRING(REPLACE "-pedantic" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

# ----------------------------------------------------
# --- DEPENDANCIES -----------------------------------
# ----------------------------------------------------
SEARCH_FOR_EIGEN()
SET(BOOST_COMPONENTS python)
SEARCH_FOR_BOOST()
set(Python_ADDITIONAL_VERSIONS 2.7)
FINDPYTHON()

# ----------------------------------------------------
# --- INCLUDE ----------------------------------------
# ----------------------------------------------------
SET(${PROJECT_NAME}_HEADERS
  include/eigenpy/eigenpy.hpp
  include/eigenpy/exception.hpp
  include/eigenpy/details.hpp
  include/eigenpy/fwd.hpp
  include/eigenpy/map.hpp
  include/eigenpy/geometry.hpp
  include/eigenpy/angle-axis.hpp
  include/eigenpy/quaternion.hpp
)

# ----------------------------------------------------
# --- TARGETS ----------------------------------------
# ----------------------------------------------------

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
# Add Boost path to include directories.
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

SET(${PROJECT_NAME}_SOURCES
  src/exception.cpp
  src/eigenpy.cpp
  src/details.cpp
  src/angle-axis.cpp
  src/quaternion.cpp
  )

ADD_LIBRARY(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES})

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
INSTALL(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)


# ----------------------------------------------------
# --- UNIT TEST --------------------------------------
# ----------------------------------------------------
ADD_LIBRARY(matrix SHARED unittest/matrix.cpp)
TARGET_LINK_LIBRARIES(matrix ${Boost_LIBRARIES} eigenpy)
SET_TARGET_PROPERTIES(matrix PROPERTIES PREFIX "")

ADD_LIBRARY(geometry SHARED unittest/geometry.cpp)
TARGET_LINK_LIBRARIES(geometry ${Boost_LIBRARIES} eigenpy)
SET_TARGET_PROPERTIES(geometry PROPERTIES PREFIX "")

PKG_CONFIG_APPEND_CFLAGS(${_Eigen_CFLAGS})
PKG_CONFIG_APPEND_CFLAGS("-I${PYTHON_INCLUDE_DIRS}")
#PKG_CONFIG_APPEND_LIBS_RAW(${Boost_LIBRARIES})
PKG_CONFIG_APPEND_LIBS("boost_python")
PKG_CONFIG_APPEND_LIBS_RAW(${PYTHON_LIBRARIES})

 PKG_CONFIG_APPEND_LIBS(${PROJECT_NAME})

SETUP_PROJECT_FINALIZE()
SETUP_PROJECT_CPACK()

