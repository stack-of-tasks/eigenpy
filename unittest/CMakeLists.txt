#
# Copyright (c) 2014-2019 CNRS
# Copyright (c) 2018-2019 INRIA
#

MACRO(ADD_LIB_UNIT_TEST test PKGS)
  IF (BUILD_UNIT_TESTS)
    ADD_LIBRARY(${test} SHARED ${test})
  ELSE (BUILD_UNIT_TESTS)
    ADD_LIBRARY(${test} SHARED EXCLUDE_FROM_ALL ${test})
  ENDIF (BUILD_UNIT_TESTS)
  FOREACH(PKG ${PKGS})
    PKG_CONFIG_USE_DEPENDENCY(${test} ${PKG})
  ENDFOREACH(PKG)
  TARGET_LINK_LIBRARIES(${test} ${PROJECT_NAME})
  TARGET_LINK_BOOST_PYTHON(${test})
  SET_TARGET_PROPERTIES(${test} PROPERTIES PREFIX "")

  IF(APPLE)
    # We need to change the extension for python bindings
    SET_TARGET_PROPERTIES(${test} PROPERTIES SUFFIX ".so")
  ENDIF(APPLE)

  ADD_TEST(NAME ${test} COMMAND ${PYTHON_EXECUTABLE} -c "import ${test}")
  ADD_DEPENDENCIES(check ${test})
ENDMACRO(ADD_LIB_UNIT_TEST)

ADD_CUSTOM_TARGET(check COMMAND ${CMAKE_CTEST_COMMAND})

ADD_LIB_UNIT_TEST(matrix "eigen3")
ADD_LIB_UNIT_TEST(geometry "eigen3")
IF(NOT ${EIGEN3_VERSION} VERSION_LESS "3.2.0")
  ADD_LIB_UNIT_TEST(ref "eigen3")
ENDIF()

ADD_PYTHON_UNIT_TEST("py-matrix" "unittest/python/test_matrix.py" "unittest")
ADD_PYTHON_UNIT_TEST("py-geometry" "unittest/python/test_geometry.py" "unittest")
ADD_PYTHON_UNIT_TEST("py-switch" "unittest/python/test_switch.py" "python/eigenpy")
ADD_PYTHON_UNIT_TEST("py-dimensions" "unittest/python/test_dimensions.py" "python/eigenpy")
