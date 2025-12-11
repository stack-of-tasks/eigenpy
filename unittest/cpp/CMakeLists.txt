# Add test that depends on the eigenpy library
# import eigenpy is not authorized in those tests
function(eigenpy_add_test name)
  jrl_boostpy_add_module(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE eigenpy)
  jrl_target_set_default_compile_options(${name} PRIVATE)
  jrl_target_enforce_msvc_conformance(${name} PRIVATE)

  jrl_target_set_output_directory(${name} OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name})

  jrl_python_generate_init_py(
      ${name}
      OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${name}/__init__.py
  )

  add_test(
    NAME "Eigenpy: import ${name}"
    COMMAND $<TARGET_FILE:Python::Interpreter> -c "import ${name}"
  )

  set(test_name "Eigenpy: test_${name}.py")
  add_test(
    NAME ${test_name}
    COMMAND $<TARGET_FILE:Python::Interpreter> ${CMAKE_CURRENT_SOURCE_DIR}/test_${name}.py
  )
  set_tests_properties(
    ${test_name}
    PROPERTIES ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:${name}>" DEPENDS ${name}
  )
endfunction()

eigenpy_add_test(matrix)
eigenpy_add_test(type_info)
eigenpy_add_test(multiple_registration)
if(BUILD_TESTING_SCIPY)
  eigenpy_add_test(sparse_matrix)
endif()
eigenpy_add_test(tensor)
eigenpy_add_test(geometry)
eigenpy_add_test(complex)
eigenpy_add_test(deprecation_policy)
eigenpy_add_test(return_by_ref)
eigenpy_add_test(include)
eigenpy_add_test(eigen_ref)
eigenpy_add_test(user_type)
eigenpy_add_test(std_vector)
eigenpy_add_test(std_array)
eigenpy_add_test(std_pair)
eigenpy_add_test(std_map)
eigenpy_add_test(user_struct)
eigenpy_add_test(std_unique_ptr)
eigenpy_add_test(virtual_factory)

set(TEST_TYPE "boost::optional")
set(MODNAME boost_optional)
configure_file(optional.cpp.in ${CMAKE_CURRENT_SOURCE_DIR}/boost_optional.cpp)
eigenpy_add_test(boost_optional)

set(TEST_TYPE "std::optional")
set(MODNAME std_optional)
configure_file(optional.cpp.in ${CMAKE_CURRENT_SOURCE_DIR}/std_optional.cpp)
eigenpy_add_test(std_optional)

set(TEST_TYPE "boost::variant")
set(MODNAME boost_variant)
configure_file(variant.cpp.in ${CMAKE_CURRENT_SOURCE_DIR}/boost_variant.cpp)
eigenpy_add_test(boost_variant)

set(TEST_TYPE "std::variant")
set(MODNAME std_variant)
configure_file(variant.cpp.in ${CMAKE_CURRENT_SOURCE_DIR}/std_variant.cpp)
eigenpy_add_test(std_variant)
