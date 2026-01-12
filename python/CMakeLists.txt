jrl_boostpy_add_module(eigenpy_pywrap bindings.cpp)
target_link_libraries(eigenpy_pywrap PRIVATE eigenpy)
jrl_target_set_default_compile_options(eigenpy_pywrap PRIVATE)
jrl_target_set_output_directory(eigenpy_pywrap OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/site-packages/eigenpy)

jrl_python_generate_init_py(
  eigenpy_pywrap OUTPUT_PATH
  ${CMAKE_BINARY_DIR}/lib/site-packages/eigenpy/__init__.py
)

if(GENERATE_PYTHON_STUBS)
  jrl_boostpy_add_stubs(
    eigenpy_pywrap_stubs
    MODULE eigenpy.eigenpy_pywrap
    OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib/site-packages
    PYTHON_PATH ${CMAKE_BINARY_DIR}/lib/site-packages
    DEPENDS eigenpy_pywrap
    VERBOSE
  )
endif()

jrl_python_compile_all(DIRECTORY ${CMAKE_BINARY_DIR}/lib/site-packages/eigenpy)

jrl_python_compute_install_dir(python_install_dir)
install(
  DIRECTORY ${CMAKE_BINARY_DIR}/lib/site-packages/eigenpy
  DESTINATION ${python_install_dir}
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "*.pyc"
  PATTERN "*.pyi"
  PATTERN "*.typed"
  PATTERN "*.so"
  PATTERN "*.pyd"
)
