#
# Copyright (c) 2014-2019 CNRS Copyright (c) 2018-2024 INRIA
#

function(ADD_LIB_UNIT_TEST test)
  create_ctest_build_tests_target()
  set(test_target ${PROJECT_NAME}-${test})

  if(BUILD_TESTING)
    add_library(${test_target} SHARED "${test}.cpp")
  else()
    add_library(${test_target} SHARED EXCLUDE_FROM_ALL "${test}.cpp")
  endif()
  set_standard_output_directory(${test_target})

  target_link_libraries(${test_target} PUBLIC ${PROJECT_NAME})
  set_target_properties(
    ${test_target}
    PROPERTIES PREFIX ""
               LIBRARY_OUTPUT_NAME ${test}
               RUNTIME_OUTPUT_NAME ${test})

  set_target_properties(${test_target} PROPERTIES SUFFIX ${PYTHON_EXT_SUFFIX})

  add_test(
    NAME ${test_target}
    COMMAND ${PYTHON_EXECUTABLE} -c "import ${test}"
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${test_target}>)

  add_dependencies(build_tests ${test_target})
  if(NOT BUILD_TESTING)
    set_tests_properties(${test_target} PROPERTIES DEPENDS ctest_build_tests)
  endif(NOT BUILD_TESTING)
endfunction()

add_dependencies(build_tests ${PYWRAP})
add_lib_unit_test(matrix)
add_lib_unit_test(type_info)
add_lib_unit_test(multiple_registration)
if(BUILD_TESTING_SCIPY)
  find_scipy()
  add_lib_unit_test(sparse_matrix)
endif()
add_lib_unit_test(tensor)
add_lib_unit_test(geometry)
add_lib_unit_test(complex)
add_lib_unit_test(deprecation_policy)
add_lib_unit_test(return_by_ref)
add_lib_unit_test(include)
if(NOT ${EIGEN3_VERSION} VERSION_LESS "3.2.0")
  add_lib_unit_test(eigen_ref)
endif()

if(NOT NUMPY_WITH_BROKEN_UFUNC_SUPPORT)
  add_lib_unit_test(user_type)
endif()
add_lib_unit_test(std_vector)
add_lib_unit_test(std_array)
add_lib_unit_test(std_pair)
add_lib_unit_test(std_map)
add_lib_unit_test(user_struct)

if(CMAKE_CXX_STANDARD GREATER 14 AND CMAKE_CXX_STANDARD LESS 98)
  add_lib_unit_test(std_unique_ptr)
endif()

function(add_python_lib_unit_test name source)
  set(test_target ${PROJECT_NAME}-${name})
  add_python_unit_test(${test_target} ${source} "lib" "bin")
endfunction()

function(add_python_eigenpy_lib_unit_test name source)
  set(test_target ${PROJECT_NAME}-${name})
  add_python_unit_test(${test_target} ${source} "lib" "bin" "python")
  set_tests_properties(${test_target} PROPERTIES DEPENDS ${PYWRAP})
endfunction()

function(add_python_eigenpy_unit_test name source)
  set(test_target ${PROJECT_NAME}-${name})
  add_python_unit_test(${test_target} ${source} "python")
  set_tests_properties(${test_target} PROPERTIES DEPENDS ${PYWRAP})
endfunction()

function(config_test test tagname opttype)
  set(MODNAME ${test}_${tagname})
  set(TEST_TYPE ${opttype})
  configure_file(${test}.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/${MODNAME}.cpp)

  set(py_file test_${test}_${tagname}.py)
  configure_file(python/test_${test}.py.in
                 ${CMAKE_CURRENT_BINARY_DIR}/python/${py_file})
  add_lib_unit_test(${MODNAME})
  set(PYTHON_TEST_NAME "${PROJECT_NAME}-py-${test}-${tagname}")
  add_test(NAME ${PYTHON_TEST_NAME}
           COMMAND ${PYTHON_EXECUTABLE}
                   "${CMAKE_CURRENT_BINARY_DIR}/python/${py_file}")
  compute_pythonpath(ENV_VARIABLES "lib" "bin")
  set_tests_properties(${PYTHON_TEST_NAME} PROPERTIES ENVIRONMENT
                                                      "${ENV_VARIABLES}")
endfunction()

config_test(variant boost "boost::variant")
if(CMAKE_CXX_STANDARD GREATER 14 AND CMAKE_CXX_STANDARD LESS 98)
  config_test(variant std "std::variant")
endif()

config_test(bind_optional boost "boost::optional")
if(CMAKE_CXX_STANDARD GREATER 14 AND CMAKE_CXX_STANDARD LESS 98)
  config_test(bind_optional std "std::optional")
endif()

add_lib_unit_test(bind_virtual_factory)

add_python_lib_unit_test("py-matrix" "unittest/python/test_matrix.py")
add_python_lib_unit_test("py-type-info" "unittest/python/test_type_info.py")
add_python_lib_unit_test("py-multiple-registration"
                         "unittest/python/test_multiple_registration.py")

add_python_lib_unit_test("py-tensor" "unittest/python/test_tensor.py")
add_python_lib_unit_test("py-Geometry" "unittest/python/test_Geometry.py")
add_python_lib_unit_test("py-complex" "unittest/python/test_complex.py")
add_python_lib_unit_test("py-deprecation-policy"
                         "unittest/python/test_deprecation_policy.py")
add_python_lib_unit_test("py-return-by-ref"
                         "unittest/python/test_return_by_ref.py")
add_python_lib_unit_test("py-eigen-ref" "unittest/python/test_eigen_ref.py")

if(NOT NUMPY_WITH_BROKEN_UFUNC_SUPPORT)
  add_python_lib_unit_test("py-user-type" "unittest/python/test_user_type.py")
endif()

add_python_eigenpy_lib_unit_test("py-dimensions"
                                 "unittest/python/test_dimensions.py")

add_python_eigenpy_lib_unit_test("py-version" "unittest/python/test_version.py")

add_python_eigenpy_lib_unit_test("py-EigenSolver"
                                 "unittest/python/test_EigenSolver.py")

add_python_eigenpy_lib_unit_test(
  "py-GeneralizedEigenSolver" "unittest/python/test_GeneralizedEigenSolver.py")

add_python_eigenpy_lib_unit_test(
  "py-GeneralizedSelfAdjointEigenSolver"
  "unittest/python/test_GeneralizedSelfAdjointEigenSolver.py")

add_python_eigenpy_lib_unit_test("py-ComplexEigenSolver"
                                 "unittest/python/test_ComplexEigenSolver.py")

add_python_eigenpy_lib_unit_test("py-ComplexSchur"
                                 "unittest/python/test_ComplexSchur.py")

add_python_eigenpy_lib_unit_test(
  "py-SelfAdjointEigenSolver" "unittest/python/test_SelfAdjointEigenSolver.py")

add_python_eigenpy_lib_unit_test(
  "py-HessenbergDecomposition"
  "unittest/python/test_HessenbergDecomposition.py")

add_python_eigenpy_lib_unit_test("py-RealQZ" "unittest/python/test_RealQZ.py")

add_python_eigenpy_lib_unit_test("py-RealSchur"
                                 "unittest/python/test_RealSchur.py")

add_python_eigenpy_lib_unit_test("py-Tridiagonalization"
                                 "unittest/python/test_Tridiagonalization.py")

add_python_eigenpy_lib_unit_test("py-LLT" "unittest/python/test_LLT.py")

add_python_eigenpy_lib_unit_test("py-LDLT" "unittest/python/test_LDLT.py")

add_python_eigenpy_lib_unit_test("py-FullPivLU"
                                 "unittest/python/test_FullPivLU.py")

add_python_eigenpy_lib_unit_test("py-PartialPivLU"
                                 "unittest/python/test_PartialPivLU.py")

add_python_eigenpy_lib_unit_test("py-id" "unittest/python/test_id.py")

add_python_eigenpy_lib_unit_test("py-QR" "unittest/python/test_QR.py")

add_python_eigenpy_lib_unit_test("py-BDCSVD" "unittest/python/test_BDCSVD.py")

add_python_eigenpy_lib_unit_test("py-JacobiSVD"
                                 "unittest/python/test_JacobiSVD.py")

if(NOT WIN32)
  add_python_eigenpy_lib_unit_test("py-MINRES" "unittest/python/test_MINRES.py")
endif(NOT WIN32)

add_python_eigenpy_lib_unit_test("py-std-vector"
                                 "unittest/python/test_std_vector.py")

add_python_lib_unit_test("py-std-array" "unittest/python/test_std_array.py")

add_python_lib_unit_test("py-std-map" "unittest/python/test_std_map.py")

add_python_lib_unit_test("py-std-pair" "unittest/python/test_std_pair.py")

add_python_lib_unit_test("py-user-struct" "unittest/python/test_user_struct.py")

if(CMAKE_CXX_STANDARD GREATER 14 AND CMAKE_CXX_STANDARD LESS 98)
  add_python_lib_unit_test("py-std-unique-ptr"
                           "unittest/python/test_std_unique_ptr.py")
endif()

add_python_lib_unit_test("py-bind-virtual"
                         "unittest/python/test_bind_virtual.py")

if(BUILD_TESTING_SCIPY)
  add_python_lib_unit_test("py-sparse-matrix"
                           "unittest/python/test_sparse_matrix.py")

  add_python_eigenpy_unit_test(
    "py-SimplicialLLT"
    "unittest/python/decompositions/sparse/test_SimplicialLLT.py")
  add_python_eigenpy_unit_test(
    "py-SimplicialLDLT"
    "unittest/python/decompositions/sparse/test_SimplicialLDLT.py")
  add_python_eigenpy_unit_test("py-IncompleteCholesky"
                               "unittest/python/test_IncompleteCholesky.py")
  add_python_eigenpy_unit_test("py-IncompleteLUT"
                               "unittest/python/test_IncompleteLUT.py")
  add_python_eigenpy_unit_test(
    "py-SparseLU" "unittest/python/decompositions/sparse/test_SparseLU.py")
  add_python_eigenpy_unit_test(
    "py-SparseQR" "unittest/python/decompositions/sparse/test_SparseQR.py")

  if(BUILD_WITH_CHOLMOD_SUPPORT)
    add_python_eigenpy_unit_test(
      "py-CholmodSimplicialLLT"
      "unittest/python/decompositions/sparse/cholmod/test_CholmodSimplicialLLT.py"
    )

    add_python_eigenpy_unit_test(
      "py-CholmodSimplicialLDLT"
      "unittest/python/decompositions/sparse/cholmod/test_CholmodSimplicialLDLT.py"
    )

    add_python_eigenpy_unit_test(
      "py-CholmodSupernodalLLT"
      "unittest/python/decompositions/sparse/cholmod/test_CholmodSupernodalLLT.py"
    )
  endif(BUILD_WITH_CHOLMOD_SUPPORT)

  if(BUILD_WITH_ACCELERATE_SUPPORT)
    add_python_eigenpy_unit_test(
      "py-Accelerate"
      "unittest/python/decompositions/sparse/test_Accelerate.py")
  endif(BUILD_WITH_ACCELERATE_SUPPORT)
endif()
