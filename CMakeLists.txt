CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "DEBUG")
ENDIF()

IF(NOT BOOST_NUMPY_DIR)
  SET(BOOST_NUMPY_DIR ".")
ENDIF()

FIND_PACKAGE(Boost 1.45.0)

IF(Boost_FOUND)

  INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}" "/usr/include/python2.7" "/usr/include/eigen3" "Boost.NumPy")
  LINK_DIRECTORIES(${BOOST_NUMPY_DIR})

  SET(Boost_USE_STATIC_LIBS OFF)
  SET(Boost_USE_MULTITHREADED ON)
  SET(Boost_USE_STATIC_RUNTIME OFF)
  FIND_PACKAGE(Boost 1.45.0 COMPONENTS python)

  ADD_LIBRARY(eigenpy SHARED unittest/libeigenpy.cpp)
  TARGET_LINK_LIBRARIES(eigenpy ${Boost_LIBRARIES})

  ADD_LIBRARY(simple SHARED unittest/alpha/simple.cpp)
  TARGET_LINK_LIBRARIES(simple ${Boost_LIBRARIES})

  ADD_LIBRARY(mystring SHARED unittest/alpha/mystring.cpp)
  TARGET_LINK_LIBRARIES(mystring ${Boost_LIBRARIES})

  ADD_LIBRARY(bnpy SHARED unittest/alpha/bnpy.cpp)
  TARGET_LINK_LIBRARIES(bnpy ${Boost_LIBRARIES})

  ADD_LIBRARY(eigen SHARED unittest/alpha/eigen.cpp)
  TARGET_LINK_LIBRARIES(eigen ${Boost_LIBRARIES} libboost_numpy.so)

  ADD_LIBRARY(eigenc SHARED unittest/alpha/eigenc.cpp)
  TARGET_LINK_LIBRARIES(eigenc ${Boost_LIBRARIES})

  ADD_LIBRARY(eigentemplate SHARED unittest/alpha/eigentemplate.cpp)
  TARGET_LINK_LIBRARIES(eigentemplate ${Boost_LIBRARIES})

ELSEIF(NOT Boost_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find correct Boost version. Did you set BOOST_ROOT?")
ENDIF()


IF(CMAKE_COMPILER_IS_GNUCXX)
  ADD_DEFINITIONS("-Wall")
ELSE()
  MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
ENDIF()
