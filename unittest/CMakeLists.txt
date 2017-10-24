# 
# Copyright (c) 2016 CNRS
# 
# This file is part of eigenpy
# eigenpy is free software: you can redistribute it
# and/or modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
# eigenpy is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Lesser Public License for more details. You should have
# received a copy of the GNU Lesser General Public License along with
# eigenpy If not, see
# <http://www.gnu.org/licenses/>.

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

