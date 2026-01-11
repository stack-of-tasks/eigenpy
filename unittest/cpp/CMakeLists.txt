# Add test that depends on the eigenpy library
# NOTE: "import eigenpy" is not authorized in those tests
function(eigenpy_add_test name)
  cmake_parse_arguments(arg "" "SOURCE_DIR" "" ${ARGN})

  if(arg_SOURCE_DIR)
    set(source_dir ${arg_SOURCE_DIR})
  else()
    set(source_dir ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  jrl_boostpy_add_module(${name} ${source_dir}/${name}.cpp)
  jrl_check_python_module_name(${name})
  target_link_libraries(${name} PRIVATE eigenpy)
  jrl_target_set_default_compile_options(${name} PRIVATE)
  jrl_target_enforce_msvc_conformance(${name} PRIVATE)

  jrl_target_set_output_directory(${name} OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name})

  jrl_python_generate_init_py(
      ${name}
      OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${name}/__init__.py
  )

  add_test(
    NAME "eigenpy: import ${name}"
    COMMAND $<TARGET_FILE:Python::Interpreter> -c "import ${name}"
  )

  set(test_name "eigenpy: test_${name}.py")
  add_test(
    NAME ${test_name}
    COMMAND $<TARGET_FILE:Python::Interpreter> ${source_dir}/test_${name}.py
  )
  set_tests_properties(
    ${test_name}
    PROPERTIES ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:${name}>/.."
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
configure_file(optional.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/boost_optional.cpp)
configure_file(test_optional.py.in ${CMAKE_CURRENT_BINARY_DIR}/test_boost_optional.py)
eigenpy_add_test(boost_optional SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(TEST_TYPE "std::optional")
set(MODNAME std_optional)
configure_file(optional.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/std_optional.cpp)
configure_file(test_optional.py.in ${CMAKE_CURRENT_BINARY_DIR}/test_std_optional.py)
eigenpy_add_test(std_optional SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(TEST_TYPE "boost::variant")
set(MODNAME boost_variant)
configure_file(variant.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/boost_variant.cpp)
configure_file(test_variant.py.in ${CMAKE_CURRENT_BINARY_DIR}/test_boost_variant.py)
eigenpy_add_test(boost_variant SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(TEST_TYPE "std::variant")
set(MODNAME std_variant)
configure_file(variant.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/std_variant.cpp)
configure_file(test_variant.py.in ${CMAKE_CURRENT_BINARY_DIR}/test_std_variant.py)
eigenpy_add_test(std_variant SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR})
