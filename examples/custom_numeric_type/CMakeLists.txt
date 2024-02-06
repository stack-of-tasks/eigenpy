# derived from eigenpy/CMakeLists.txt for this example

cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME eigenpy_example_custom_numeric_type)
set(PROJECT_DESCRIPTION
    "An example of using eigenpy with a custom numeric type: Boost.Multiprecision complex numbers"
)
set(PROJECT_URL
    "http://github.com/stack-of-tasks/eigenpy/examples/custom_numeric_type")
set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(PROJECT_USE_KEYWORD_LINK_LIBRARIES TRUE)
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_COMPATIBILITY_VERSION AnyNewerVersion)

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  if(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    message(
      FATAL_ERROR
        "\nPlease run the following command first:\ngit submodule update --init\n"
    )
  else()
    message(STATUS "JRL cmakemodules not found. Let's fetch it.")
    include(FetchContent)
    FetchContent_Declare(
      "jrl-cmakemodules"
      GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  endif()
endif()

# Disable -Werror on Unix for now.
set(CXX_DISABLE_WERROR True)
set(CMAKE_VERBOSE_MAKEFILE True)

option(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)
option(SUFFIX_SO_VERSION "Suffix library name with its version" OFF)

if(DEFINED BUILD_UNIT_TESTS)
  message(
    AUTHOR_WARNING "BUILD_UNIT_TESTS is deprecated. Use BUILD_TESTING instead.")
  set(BUILD_TESTING ${BUILD_UNIT_TESTS})
endif(DEFINED BUILD_UNIT_TESTS)

include("${JRL_CMAKE_MODULES}/base.cmake")
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

include("${JRL_CMAKE_MODULES}/boost.cmake")
include("${JRL_CMAKE_MODULES}/python.cmake")
include("${JRL_CMAKE_MODULES}/ide.cmake")
include("${JRL_CMAKE_MODULES}/apple.cmake")

option(GENERATE_PYTHON_STUBS
       "Generate the Python stubs associated to the Python library" OFF)

string(REPLACE "-pedantic" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

# If needed, fix CMake policy for APPLE systems
apply_default_apple_configuration()
check_minimal_cxx_standard(11 ENFORCE)

if(WIN32)
  set(LINK copy_if_different)
else(WIN32)
  set(LINK create_symlink)
endif(WIN32)

if(CMAKE_CROSSCOMPILING)
  set(PYTHON_COMPONENTS Interpreter NumPy)
else()
  set(PYTHON_COMPONENTS Interpreter Development.Module NumPy)
endif()
set(PYTHON_EXPORT_DEPENDENCY ON)
findpython(REQUIRED)

if(${NUMPY_VERSION} VERSION_LESS "1.16.0")
  set(NUMPY_WITH_BROKEN_UFUNC_SUPPORT TRUE)
endif()

if(WIN32)
  link_directories(${PYTHON_LIBRARY_DIRS})
  # # Set default Windows build paths SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY
  # ${PROJECT_BINARY_DIR}/Bin CACHE PATH "Single directory for all libraries")
  # SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/Bin CACHE PATH
  # "Single directory for all executables") SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  # ${PROJECT_BINARY_DIR}/Bin CACHE PATH "Sing$le directory for all archives")
endif(WIN32)

# ----------------------------------------------------
# --- DEPENDENCIES -----------------------------------
# ----------------------------------------------------

add_project_dependency(Eigen3 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.0.5")
add_project_dependency(eigenpy REQUIRED PKG_CONFIG_REQUIRES "eigenpy >= 3.0.0")

set_boost_default_options()
export_boost_default_options()
find_package(Boost REQUIRED)
search_for_boost_python(REQUIRED)

# ----------------------------------------------------
# --- INCLUDE ----------------------------------------
# ----------------------------------------------------
set(${PROJECT_NAME}_HEADERS include/header.hpp)

set(${PROJECT_NAME}_SOURCES src/src.cpp)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES}
                                   ${${PROJECT_NAME}_HEADERS})

target_include_directories(
  ${PROJECT_NAME} SYSTEM
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:include>)

target_link_directories(${PROJECT_NAME}
  PUBLIC ${EIGENPY_LIB_DIR})

modernize_target_link_libraries(
  ${PROJECT_NAME}
  SCOPE
  PUBLIC
  TARGETS
  Python${PYTHON_VERSION_MAJOR}::NumPy
  Eigen3::Eigen
  eigenpy
  INCLUDE_DIRS
  ${EIGEN3_INCLUDE_DIR}
  ${EIGENPY_INCLUDE_DIR}
  ${NUMPY_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIR})


target_link_libraries(${PROJECT_NAME} PUBLIC eigenpy::eigenpy mpc mpfr)

if(SUFFIX_SO_VERSION)
  set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION})
endif(SUFFIX_SO_VERSION)


#silviana did this
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".so")

if(NOT WIN32)
  target_compile_options(
    ${PROJECT_NAME} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:-bigobj>
                            "-Wno-conversion")
else()
  target_compile_options(${PROJECT_NAME}
                         PRIVATE $<$<CXX_COMPILER_ID:MSVC>:-bigobj>)
  target_compile_definitions(${PROJECT_NAME} PUBLIC "HAVE_SNPRINTF")
endif()

target_link_boost_python(${PROJECT_NAME} PUBLIC)



install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${TARGETS_EXPORT_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_header_group(${PROJECT_NAME}_HEADERS)
add_source_group(${PROJECT_NAME}_SOURCES)

# Install package for ROS
install(FILES package.xml DESTINATION share/example_)
# Allows Colcon to find non-Ament packages when using workspace underlays
file(
  WRITE
  ${CMAKE_CURRENT_BINARY_DIR}/share/ament_index/resource_index/packages/${PROJECT_NAME}
  "")
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/share/ament_index/resource_index/packages/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages)
file(
  WRITE
  ${CMAKE_CURRENT_BINARY_DIR}/share/${PROJECT_NAME}/hook/ament_prefix_path.dsv
  "prepend-non-duplicate;AMENT_PREFIX_PATH;")
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/share/${PROJECT_NAME}/hook/ament_prefix_path.dsv
  DESTINATION share/${PROJECT_NAME}/hook)
file(WRITE
     ${CMAKE_CURRENT_BINARY_DIR}/share/${PROJECT_NAME}/hook/python_path.dsv
     "prepend-non-duplicate;PYTHONPATH;${PYTHON_SITELIB}")
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/share/${PROJECT_NAME}/hook/python_path.dsv
  DESTINATION share/${PROJECT_NAME}/hook)

# ----------------------------------------------------
# --- PYTHON LIBRARY ---------------------------------
# ----------------------------------------------------
# add_subdirectory(python)

pkg_config_append_libs(${PROJECT_NAME})
pkg_config_append_cflags("-I${PYTHON_INCLUDE_DIRS}")
pkg_config_append_cflags("-I${NUMPY_INCLUDE_DIRS}")
pkg_config_append_boost_libs(${BOOST_COMPONENTS})
