#
# Copyright (c) 2014-2019 CNRS
# Copyright (c) 2018-2021 INRIA
#

MACRO(ADD_LIB_UNIT_TEST test)
  CREATE_CTEST_BUILD_TESTS_TARGET()

  IF(BUILD_TESTING)
    ADD_LIBRARY(${test} SHARED "${test}.cpp")
  ELSE(BUILD_TESTING)
    ADD_LIBRARY(${test} SHARED EXCLUDE_FROM_ALL "${test}.cpp")
  ENDIF(BUILD_TESTING)

  TARGET_LINK_LIBRARIES(${test} PUBLIC ${PROJECT_NAME})
  SET_TARGET_PROPERTIES(${test} PROPERTIES PREFIX "")

  SET_TARGET_PROPERTIES(${test} PROPERTIES SUFFIX ${PYTHON_EXT_SUFFIX})

  ADD_TEST(NAME ${test} COMMAND ${PYTHON_EXECUTABLE} -c "import ${test}")

  ADD_DEPENDENCIES(build_tests ${test})
  IF(NOT BUILD_TESTING)
    SET_TESTS_PROPERTIES(${test} PROPERTIES DEPENDS ctest_build_tests)
  ENDIF(NOT BUILD_TESTING)
ENDMACRO(ADD_LIB_UNIT_TEST)

ADD_LIB_UNIT_TEST(matrix)
ADD_LIB_UNIT_TEST(geometry)
ADD_LIB_UNIT_TEST(complex)
ADD_LIB_UNIT_TEST(return_by_ref)
ADD_LIB_UNIT_TEST(include)
IF(NOT ${EIGEN3_VERSION} VERSION_LESS "3.2.0")
  ADD_LIB_UNIT_TEST(eigen_ref)
ENDIF()
ADD_LIB_UNIT_TEST(user_type)

ADD_PYTHON_UNIT_TEST("py-matrix" "unittest/python/test_matrix.py" "unittest")
ADD_PYTHON_UNIT_TEST("py-geometry" "unittest/python/test_geometry.py" "unittest")
ADD_PYTHON_UNIT_TEST("py-complex" "unittest/python/test_complex.py" "unittest")
ADD_PYTHON_UNIT_TEST("py-return-by-ref" "unittest/python/test_return_by_ref.py" "unittest")
ADD_PYTHON_UNIT_TEST("py-eigen-ref" "unittest/python/test_eigen_ref.py" "unittest")
ADD_PYTHON_UNIT_TEST("py-user-type" "unittest/python/test_user_type.py" "unittest")

ADD_PYTHON_UNIT_TEST("py-switch" "unittest/python/test_switch.py" "python;unittest")
SET_TESTS_PROPERTIES("py-switch" PROPERTIES DEPENDS ${PYWRAP})

ADD_PYTHON_UNIT_TEST("py-dimensions" "unittest/python/test_dimensions.py" "python;unittest")
SET_TESTS_PROPERTIES("py-dimensions" PROPERTIES DEPENDS ${PYWRAP})

ADD_PYTHON_UNIT_TEST("py-version" "unittest/python/test_version.py" "python;unittest")
SET_TESTS_PROPERTIES("py-version" PROPERTIES DEPENDS ${PYWRAP})

ADD_PYTHON_UNIT_TEST("py-eigen-solver" "unittest/python/test_eigen_solver.py" "python;unittest")
SET_TESTS_PROPERTIES("py-eigen-solver" PROPERTIES DEPENDS ${PYWRAP})

ADD_PYTHON_UNIT_TEST("py-self-adjoint-eigen-solver" "unittest/python/test_self_adjoint_eigen_solver.py" "python;unittest")
SET_TESTS_PROPERTIES("py-self-adjoint-eigen-solver" PROPERTIES DEPENDS ${PYWRAP})

ADD_PYTHON_UNIT_TEST("py-LLT" "unittest/python/test_LLT.py" "python;unittest")
SET_TESTS_PROPERTIES("py-LLT" PROPERTIES DEPENDS ${PYWRAP})

ADD_PYTHON_UNIT_TEST("py-LDLT" "unittest/python/test_LDLT.py" "python;unittest")
SET_TESTS_PROPERTIES("py-LDLT" PROPERTIES DEPENDS ${PYWRAP})

ADD_PYTHON_UNIT_TEST("py-MINRES" "unittest/python/test_MINRES.py" "python;unittest")
SET_TESTS_PROPERTIES("py-MINRES" PROPERTIES DEPENDS ${PYWRAP})
