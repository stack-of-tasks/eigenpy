#
# Copyright (c) 2014-2019 CNRS Copyright (c) 2018-2023 INRIA
#

macro(ADD_LIB_UNIT_TEST test)
  create_ctest_build_tests_target()

  if(BUILD_TESTING)
    add_library(${test} SHARED "${test}.cpp")
  else(BUILD_TESTING)
    add_library(${test} SHARED EXCLUDE_FROM_ALL "${test}.cpp")
  endif(BUILD_TESTING)

  target_link_libraries(${test} PUBLIC ${PROJECT_NAME})
  set_target_properties(${test} PROPERTIES PREFIX "")

  set_target_properties(${test} PROPERTIES SUFFIX ${PYTHON_EXT_SUFFIX})

  add_test(NAME ${test} COMMAND ${PYTHON_EXECUTABLE} -c "import ${test}")

  add_dependencies(build_tests ${test})
  if(NOT BUILD_TESTING)
    set_tests_properties(${test} PROPERTIES DEPENDS ctest_build_tests)
  endif(NOT BUILD_TESTING)
endmacro(ADD_LIB_UNIT_TEST)

add_lib_unit_test(matrix)
add_lib_unit_test(tensor)
add_lib_unit_test(geometry)
add_lib_unit_test(complex)
add_lib_unit_test(return_by_ref)
add_lib_unit_test(include)
if(NOT ${EIGEN3_VERSION} VERSION_LESS "3.2.0")
  add_lib_unit_test(eigen_ref)
endif()

if(NOT NUMPY_WITH_BROKEN_UFUNC_SUPPORT)
  add_lib_unit_test(user_type)
endif()
add_lib_unit_test(std_vector)
add_lib_unit_test(user_struct)

function(config_bind_optional tagname opttype)
  set(MODNAME bind_optional_${tagname})
  set(OPTIONAL ${opttype})
  configure_file(bind_optional.cpp.in ${MODNAME}.cpp)

  set(py_file test_optional_${tagname}.py)
  configure_file(python/test_optional.py.in
                 ${CMAKE_CURRENT_BINARY_DIR}/python/${py_file})
  add_lib_unit_test(${MODNAME})
  add_python_unit_test("py-optional-${tagname}" "unittest/python/${py_file}"
                       "unittest")
endfunction()

config_bind_optional(boost "boost::optional")
if(CMAKE_CXX_STANDARD GREATER 14 AND CMAKE_CXX_STANDARD LESS 98)
  config_bind_optional(std "std::optional")
endif()

add_lib_unit_test(bind_virtual_factory)

add_python_unit_test("py-matrix" "unittest/python/test_matrix.py" "unittest")

add_python_unit_test("py-tensor" "unittest/python/test_tensor.py" "unittest")
add_python_unit_test("py-geometry" "unittest/python/test_geometry.py"
                     "unittest")
add_python_unit_test("py-complex" "unittest/python/test_complex.py" "unittest")
add_python_unit_test("py-return-by-ref" "unittest/python/test_return_by_ref.py"
                     "unittest")
add_python_unit_test("py-eigen-ref" "unittest/python/test_eigen_ref.py"
                     "unittest")

if(NOT NUMPY_WITH_BROKEN_UFUNC_SUPPORT)
  add_python_unit_test("py-user-type" "unittest/python/test_user_type.py"
                       "unittest")
endif()

add_python_unit_test("py-dimensions" "unittest/python/test_dimensions.py"
                     "python;unittest")
set_tests_properties("py-dimensions" PROPERTIES DEPENDS ${PYWRAP})

add_python_unit_test("py-version" "unittest/python/test_version.py"
                     "python;unittest")
set_tests_properties("py-version" PROPERTIES DEPENDS ${PYWRAP})

add_python_unit_test("py-eigen-solver" "unittest/python/test_eigen_solver.py"
                     "python;unittest")
set_tests_properties("py-eigen-solver" PROPERTIES DEPENDS ${PYWRAP})

add_python_unit_test(
  "py-self-adjoint-eigen-solver"
  "unittest/python/test_self_adjoint_eigen_solver.py" "python;unittest")
set_tests_properties("py-self-adjoint-eigen-solver" PROPERTIES DEPENDS
                                                               ${PYWRAP})

add_python_unit_test("py-LLT" "unittest/python/test_LLT.py" "python;unittest")
set_tests_properties("py-LLT" PROPERTIES DEPENDS ${PYWRAP})

add_python_unit_test("py-LDLT" "unittest/python/test_LDLT.py" "python;unittest")
set_tests_properties("py-LDLT" PROPERTIES DEPENDS ${PYWRAP})

if(NOT WIN32)
  add_python_unit_test("py-MINRES" "unittest/python/test_MINRES.py"
                       "python;unittest")
  set_tests_properties("py-MINRES" PROPERTIES DEPENDS ${PYWRAP})
endif(NOT WIN32)

add_python_unit_test("py-std-vector" "unittest/python/test_std_vector.py"
                     "python;unittest")
set_tests_properties("py-std-vector" PROPERTIES DEPENDS ${PYWRAP})

add_python_unit_test("py-user-struct" "unittest/python/test_user_struct.py"
                     "python;unittest")
set_tests_properties("py-user-struct" PROPERTIES DEPENDS ${PYWRAP})

add_python_unit_test("py-bind-virtual" "unittest/python/test_bind_virtual.py"
                     "python;unittest")
set_tests_properties("py-bind-virtual" PROPERTIES DEPENDS ${PYWRAP})
