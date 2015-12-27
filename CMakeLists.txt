#
#  Copyright 2014 CNRS
#

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)
INCLUDE(cmake/python.cmake)

SET(PROJECT_NAME eigenpy)
SET(PROJECT_DESCRIPTION "Wrapping Eigen3 -- numpy")
SET(PROJECT_URL "http://github.com/stack-of-tasks/eigenpy")

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE True)

SETUP_PROJECT()
STRING(REPLACE "-pedantic" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

IF(APPLE)
  SET(CMAKE_MACOSX_RPATH TRUE)
ENDIF(APPLE)

# ----------------------------------------------------
# --- OPTIONS  ---------------------------------------
# ----------------------------------------------------
OPTION (EIGEN_NUMPY_ALIGNED "Directly aligned data between Numpy and Eigen" OFF)

IF(EIGEN_NUMPY_ALIGNED)
  ADD_DEFINITIONS(-DEIGENPY_ALIGNED)
ENDIF(EIGEN_NUMPY_ALIGNED)

# ----------------------------------------------------
# --- DEPENDANCIES -----------------------------------
# ----------------------------------------------------
ADD_REQUIRED_DEPENDENCY("eigen3 >= 3.0.5")
SET(BOOST_COMPONENTS python)
SEARCH_FOR_BOOST()
set(Python_ADDITIONAL_VERSIONS 2.7)
FINDPYTHON()
FIND_NUMPY()

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
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})
# Add Boost path to include directories.
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

SET(${PROJECT_NAME}_SOURCES
  src/exception.cpp
  src/eigenpy.cpp
  src/details.cpp
  src/angle-axis.cpp
  src/quaternion.cpp
)

# With Darwin system, the library needs to be static for a complete exposion of eigen structure under Python
IF(APPLE)
  ADD_LIBRARY(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
ELSE(APPLE)
  ADD_LIBRARY(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
ENDIF(APPLE)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
PKG_CONFIG_USE_DEPENDENCY(${PROJECT_NAME} eigen3)
INSTALL(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

# ----------------------------------------------------
# --- UNIT TEST --------------------------------------
# ----------------------------------------------------
ADD_LIBRARY(matrix SHARED unittest/matrix.cpp)
TARGET_LINK_LIBRARIES(matrix ${Boost_LIBRARIES} ${PROJECT_NAME})
SET_TARGET_PROPERTIES(matrix PROPERTIES PREFIX "")

ADD_LIBRARY(geometry SHARED unittest/geometry.cpp)
TARGET_LINK_LIBRARIES(geometry ${Boost_LIBRARIES} ${PROJECT_NAME})
SET_TARGET_PROPERTIES(geometry PROPERTIES PREFIX "")

IF(EIGEN_NUMPY_ALIGNED)
  PKG_CONFIG_APPEND_CFLAGS("-DEIGENPY_ALIGNED")
ENDIF(EIGEN_NUMPY_ALIGNED)
PKG_CONFIG_APPEND_CFLAGS("-I${PYTHON_INCLUDE_DIRS}")
PKG_CONFIG_APPEND_CFLAGS("-I${NUMPY_INCLUDE_DIRS}")
PKG_CONFIG_APPEND_LIBS("boost_python")
PKG_CONFIG_APPEND_LIBS_RAW(${PYTHON_LIBRARIES})
PKG_CONFIG_APPEND_LIBS(${PROJECT_NAME})

SETUP_PROJECT_FINALIZE()
