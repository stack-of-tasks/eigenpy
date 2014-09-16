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

project(${PROJECT_NAME})
SETUP_PROJECT()
STRING(REPLACE "-pedantic" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
MESSAGE(STATUS "CFLAGS = "${CMAKE_CXX_FLAGS})

# ----------------------------------------------------
# --- DEPENDANCIES -----------------------------------
# ----------------------------------------------------
SEARCH_FOR_EIGEN()
SET(BOOST_COMPONENTS python)
SEARCH_FOR_BOOST()
set(Python_ADDITIONAL_VERSIONS 2.7)
FINDPYTHON()
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

# ----------------------------------------------------
# --- INCLUDE ----------------------------------------
# ----------------------------------------------------
SET(${PROJECT_NAME}_HEADERS
  src/eigenpy.hpp
  src/exception.hpp
  src/details.hpp
  src/fwd.hpp
  src/map.hpp
  src/geometry.hpp
  src/angle-axis.hpp
  src/quaternion.hpp
  # Deprec
  src/simple.hpp
)
MAKE_DIRECTORY("${${PROJECT_NAME}_BINARY_DIR}/include/eigenpy")
INCLUDE_DIRECTORIES(${${PROJECT_NAME}_BINARY_DIR}/include/eigenpy)

FOREACH(header ${${PROJECT_NAME}_HEADERS})
  GET_FILENAME_COMPONENT(headerName ${header} NAME)
  IF(WIN32)
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${${PROJECT_NAME}_SOURCE_DIR}/${header}
                    ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/)
  ELSE(WIN32)
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                    ${${PROJECT_NAME}_SOURCE_DIR}/${header}
                    ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/${headerName})
  ENDIF(WIN32)
  INSTALL(FILES ${${PROJECT_NAME}_SOURCE_DIR}/${header}
	  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
ENDFOREACH(header)

# ----------------------------------------------------
# --- TARGETS ----------------------------------------
# ----------------------------------------------------

ADD_LIBRARY(eigenpy SHARED
  src/exception.cpp
  src/eigenpy.cpp
  src/details.cpp
  src/angle-axis.cpp
  src/quaternion.cpp
  )
TARGET_LINK_LIBRARIES(eigenpy ${Boost_LIBRARIES})
INSTALL(TARGETS eigenpy DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)


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
PKG_CONFIG_APPEND_LIBS_RAW(${Boost_LIBRARIES})

SETUP_PROJECT_FINALIZE()
SETUP_PROJECT_CPACK()

