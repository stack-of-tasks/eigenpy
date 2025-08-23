#
# Copyright (c) 2014-2019 CNRS Copyright (c) 2018-2023 INRIA
#

cmake_minimum_required(VERSION 3.22)

set(PROJECT_NAME eigenpy)
set(PROJECT_DESCRIPTION "Bindings between Numpy and Eigen using Boost.Python")
set(PROJECT_URL "http://github.com/stack-of-tasks/eigenpy")
set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(PROJECT_USE_KEYWORD_LINK_LIBRARIES TRUE)
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_COMPATIBILITY_VERSION AnyNewerVersion)
# To enable jrl-cmakemodules compatibility with workspace we must define the two
# following lines
set(PROJECT_AUTO_RUN_FINALIZE FALSE)
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(EXISTS "${JRL_CMAKE_MODULES}/base.cmake")
  message(STATUS "JRL cmakemodules found in 'cmake/' git submodule")
else()
  find_package(jrl-cmakemodules QUIET CONFIG)
  if(jrl-cmakemodules_FOUND)
    get_property(
      JRL_CMAKE_MODULES
      TARGET jrl-cmakemodules::jrl-cmakemodules
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "JRL cmakemodules found on system at ${JRL_CMAKE_MODULES}")
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

function(set_standard_output_directory target)
  set_target_properties(
    ${target}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
               LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
               ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
endfunction()

# Disable -Werror on Unix for now.
set(CXX_DISABLE_WERROR True)
set(CMAKE_VERBOSE_MAKEFILE True)

# ----------------------------------------------------
# --- OPTIONS  ---------------------------------------
# Need to be set before including base.cmake
# ----------------------------------------------------
option(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)
option(SUFFIX_SO_VERSION "Suffix library name with its version" OFF)
option(BUILD_TESTING_SCIPY
       "Build the SciPy tests (scipy should be installed on the machine)" ON)

# ----------------------------------------------------
# --- Policy -----------------------------------------
# CMake Policy setup
# ----------------------------------------------------
# Policy can be removed when cmake_minimum_required is updated.

# We also set CMAKE_POLICY_DEFAULT_CMPXXXX because CMake modules can reset
# policy and redefine some macros like `find_dependency` that will not use our
# policy.

# Use BoostConfig module distributed by boost library instead of using FindBoost
# module distributed by CMake (to remove in 3.30).
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0167 NEW)
endif()
# install() DESTINATION paths are normalized (to remove in 3.31).
if(POLICY CMP0177)
  cmake_policy(SET CMP0177 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0177 NEW)
endif()
include("${JRL_CMAKE_MODULES}/base.cmake")
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

include("${JRL_CMAKE_MODULES}/boost.cmake")
include("${JRL_CMAKE_MODULES}/python.cmake")
include("${JRL_CMAKE_MODULES}/ide.cmake")
include("${JRL_CMAKE_MODULES}/apple.cmake")

option(GENERATE_PYTHON_STUBS
       "Generate the Python stubs associated to the Python library" OFF)

option(BUILD_WITH_CHOLMOD_SUPPORT "Build EigenPy with the Cholmod support" OFF)

if(APPLE)
  option(BUILD_WITH_ACCELERATE_SUPPORT
         "Build EigenPy with the Accelerate support" OFF)
else(APPLE)
  set(BUILD_WITH_ACCELERATE_SUPPORT FALSE)
endif(APPLE)

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

set_boost_default_options()
export_boost_default_options()
find_package(Boost REQUIRED)
search_for_boost_python(REQUIRED)

if(BUILD_WITH_CHOLMOD_SUPPORT)
  set(CMAKE_MODULE_PATH ${JRL_CMAKE_MODULES}/find-external/CHOLMOD
                        ${CMAKE_MODULE_PATH})
  add_project_dependency(CHOLMOD REQUIRED FIND_EXTERNAL "CHOLMOD")
  message(
    STATUS
      "Build with CHOLDOD support (LGPL). See CHOLMOD/Doc/License.txt for further details."
  )
  add_definitions(-DEIGENPY_WITH_CHOLMOD_SUPPORT)
endif(BUILD_WITH_CHOLMOD_SUPPORT)

if(BUILD_WITH_ACCELERATE_SUPPORT)
  if(NOT ${Eigen3_VERSION} VERSION_GREATER_EQUAL "3.4.90")
    message(
      FATAL_ERROR
        "Your version of Eigen is too low. Should be at least 3.4.90. Current version is ${Eigen3_VERSION}."
    )
  endif()

  set(CMAKE_MODULE_PATH ${JRL_CMAKE_MODULES}/find-external/Accelerate
                        ${CMAKE_MODULE_PATH})
  find_package(
    Accelerate REQUIRED # FIND_EXTERNAL "Accelerate" # We don't export yet as
                        # there might be an issue on AMR64 platforms
  )
  message(STATUS "Build with Accelerate support framework.")
  add_definitions(-DEIGENPY_WITH_ACCELERATE_SUPPORT)
endif(BUILD_WITH_ACCELERATE_SUPPORT)

# ----------------------------------------------------
# --- INCLUDE ----------------------------------------
# ----------------------------------------------------
set(${PROJECT_NAME}_UTILS_HEADERS
    include/eigenpy/utils/scalar-name.hpp
    include/eigenpy/utils/is-approx.hpp
    include/eigenpy/utils/is-aligned.hpp
    include/eigenpy/utils/traits.hpp
    include/eigenpy/utils/python-compat.hpp
    include/eigenpy/utils/empty-visitor.hpp)

set(${PROJECT_NAME}_SOLVERS_HEADERS
    include/eigenpy/solvers/solvers.hpp
    include/eigenpy/solvers/preconditioners.hpp
    include/eigenpy/solvers/IterativeSolverBase.hpp
    include/eigenpy/solvers/LeastSquaresConjugateGradient.hpp
    include/eigenpy/solvers/BiCGSTAB.hpp
    include/eigenpy/solvers/MINRES.hpp
    include/eigenpy/solvers/ConjugateGradient.hpp
    include/eigenpy/solvers/SparseSolverBase.hpp
    include/eigenpy/solvers/BasicPreconditioners.hpp
    include/eigenpy/solvers/BFGSPreconditioners.hpp
    include/eigenpy/solvers/IncompleteCholesky.hpp
    include/eigenpy/solvers/IncompleteLUT.hpp)

set(${PROJECT_NAME}_EIGEN_HEADERS include/eigenpy/eigen/EigenBase.hpp)

set(${PROJECT_NAME}_DECOMPOSITIONS_SPARSE_CHOLMOD_HEADERS
    include/eigenpy/decompositions/sparse/cholmod/CholmodBase.hpp
    include/eigenpy/decompositions/sparse/cholmod/CholmodDecomposition.hpp
    include/eigenpy/decompositions/sparse/cholmod/CholmodSimplicialLDLT.hpp
    include/eigenpy/decompositions/sparse/cholmod/CholmodSimplicialLLT.hpp
    include/eigenpy/decompositions/sparse/cholmod/CholmodSupernodalLLT.hpp)

set(${PROJECT_NAME}_DECOMPOSITIONS_SPARSE_ACCELERATE_HEADERS
    include/eigenpy/decompositions/sparse/accelerate/Accelerate.hpp
    include/eigenpy/decompositions/sparse/accelerate/accelerate.hpp)

set(${PROJECT_NAME}_DECOMPOSITIONS_SPARSE_HEADERS
    include/eigenpy/decompositions/sparse/SimplicialLLT.hpp
    include/eigenpy/decompositions/sparse/SimplicialLDLT.hpp
    include/eigenpy/decompositions/sparse/SparseLU.hpp
    include/eigenpy/decompositions/sparse/SparseQR.hpp
    include/eigenpy/decompositions/sparse/SimplicialCholesky.hpp
    include/eigenpy/decompositions/sparse/SparseSolverBase.hpp
    include/eigenpy/decompositions/sparse/LDLT.hpp
    include/eigenpy/decompositions/sparse/LLT.hpp)

if(BUILD_WITH_CHOLMOD_SUPPORT)
  list(APPEND ${PROJECT_NAME}_DECOMPOSITIONS_SPARSE_HEADERS
       ${${PROJECT_NAME}_DECOMPOSITIONS_SPARSE_CHOLMOD_HEADERS})
endif(BUILD_WITH_CHOLMOD_SUPPORT)

if(BUILD_WITH_ACCELERATE_SUPPORT)
  list(APPEND ${PROJECT_NAME}_DECOMPOSITIONS_SPARSE_HEADERS
       ${${PROJECT_NAME}_DECOMPOSITIONS_SPARSE_ACCELERATE_HEADERS})
endif(BUILD_WITH_ACCELERATE_SUPPORT)

set(${PROJECT_NAME}_DECOMPOSITIONS_HEADERS
    ${${PROJECT_NAME}_DECOMPOSITIONS_SPARSE_HEADERS}
    include/eigenpy/decompositions/decompositions.hpp
    include/eigenpy/decompositions/EigenSolver.hpp
    include/eigenpy/decompositions/GeneralizedEigenSolver.hpp
    include/eigenpy/decompositions/GeneralizedSelfAdjointEigenSolver.hpp
    include/eigenpy/decompositions/HessenbergDecomposition.hpp
    include/eigenpy/decompositions/RealQZ.hpp
    include/eigenpy/decompositions/Tridiagonalization.hpp
    include/eigenpy/decompositions/RealSchur.hpp
    include/eigenpy/decompositions/ComplexEigenSolver.hpp
    include/eigenpy/decompositions/ComplexSchur.hpp
    include/eigenpy/decompositions/FullPivLU.hpp
    include/eigenpy/decompositions/PartialPivLU.hpp
    include/eigenpy/decompositions/PermutationMatrix.hpp
    include/eigenpy/decompositions/LDLT.hpp
    include/eigenpy/decompositions/LLT.hpp
    include/eigenpy/decompositions/QR.hpp
    include/eigenpy/decompositions/HouseholderQR.hpp
    include/eigenpy/decompositions/ColPivHouseholderQR.hpp
    include/eigenpy/decompositions/CompleteOrthogonalDecomposition.hpp
    include/eigenpy/decompositions/FullPivHouseholderQR.hpp
    include/eigenpy/decompositions/SelfAdjointEigenSolver.hpp
    include/eigenpy/decompositions/SVDBase.hpp
    include/eigenpy/decompositions/BDCSVD.hpp
    include/eigenpy/decompositions/JacobiSVD.hpp
    include/eigenpy/decompositions/minres.hpp)

set(${PROJECT_NAME}_HEADERS
    ${${PROJECT_NAME}_UTILS_HEADERS}
    ${${PROJECT_NAME}_SOLVERS_HEADERS}
    ${${PROJECT_NAME}_EIGEN_HEADERS}
    ${${PROJECT_NAME}_DECOMPOSITIONS_HEADERS}
    include/eigenpy/alignment.hpp
    include/eigenpy/computation-info.hpp
    include/eigenpy/deprecation-policy.hpp
    include/eigenpy/eigenpy.hpp
    include/eigenpy/exception.hpp
    include/eigenpy/scalar-conversion.hpp
    include/eigenpy/expose.hpp
    include/eigenpy/copyable.hpp
    include/eigenpy/details.hpp
    include/eigenpy/fwd.hpp
    include/eigenpy/eigen-allocator.hpp
    include/eigenpy/eigen-to-python.hpp
    include/eigenpy/eigen-from-python.hpp
    include/eigenpy/eigen-typedef.hpp
    include/eigenpy/id.hpp
    include/eigenpy/numpy-map.hpp
    include/eigenpy/geometry.hpp
    include/eigenpy/geometry-conversion.hpp
    include/eigenpy/map.hpp
    include/eigenpy/memory.hpp
    include/eigenpy/numpy.hpp
    include/eigenpy/numpy-allocator.hpp
    include/eigenpy/numpy-type.hpp
    include/eigenpy/registration.hpp
    include/eigenpy/registration_class.hpp
    include/eigenpy/angle-axis.hpp
    include/eigenpy/quaternion.hpp
    include/eigenpy/user-type.hpp
    include/eigenpy/ufunc.hpp
    include/eigenpy/register.hpp
    include/eigenpy/std-array.hpp
    include/eigenpy/std-map.hpp
    include/eigenpy/std-pair.hpp
    include/eigenpy/std-vector.hpp
    include/eigenpy/optional.hpp
    include/eigenpy/pickle-vector.hpp
    include/eigenpy/stride.hpp
    include/eigenpy/tensor/eigen-from-python.hpp
    include/eigenpy/sparse/eigen-from-python.hpp
    include/eigenpy/scipy-allocator.hpp
    include/eigenpy/scipy-type.hpp
    include/eigenpy/variant.hpp
    include/eigenpy/std-unique-ptr.hpp
    include/eigenpy/swig.hpp
    include/eigenpy/type_info.hpp
    include/eigenpy/version.hpp)

list(
  APPEND
  ${PROJECT_NAME}_HEADERS
  ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/config.hpp
  ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/deprecated.hpp
  ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/warning.hpp)

# ----------------------------------------------------
# --- TARGETS ----------------------------------------
# ----------------------------------------------------
set(${PROJECT_NAME}_SOLVERS_SOURCES
    src/solvers/preconditioners.cpp
    src/solvers/solvers.cpp
    src/solvers/minres.cpp
    src/solvers/bicgstab.cpp
    src/solvers/conjugate-gradient.cpp
    src/solvers/least-squares-conjugate-gradient.cpp
    src/solvers/incomplete-cholesky.cpp
    src/solvers/incomplete-lut.cpp)

set(${PROJECT_NAME}_DECOMPOSITIONS_SOURCES
    src/decompositions/decompositions.cpp
    src/decompositions/eigen-solver.cpp
    src/decompositions/generalized-eigen-solver.cpp
    src/decompositions/generalized-self-adjoint-eigen-solver.cpp
    src/decompositions/complex-eigen-solver.cpp
    src/decompositions/complex-schur.cpp
    src/decompositions/llt-solver.cpp
    src/decompositions/ldlt-solver.cpp
    src/decompositions/bdcsvd-solver.cpp
    src/decompositions/jacobisvd-solver.cpp
    src/decompositions/fullpivlu-solver.cpp
    src/decompositions/hessenberg-decomposition.cpp
    src/decompositions/real-qz.cpp
    src/decompositions/tridiagonalization.cpp
    src/decompositions/real-schur.cpp
    src/decompositions/partialpivlu-solver.cpp
    src/decompositions/sparse-lu-solver.cpp
    src/decompositions/sparse-qr-solver.cpp
    src/decompositions/qr-solvers.cpp
    src/decompositions/self-adjoint-eigen-solver.cpp
    src/decompositions/permutation-matrix.cpp
    src/decompositions/simplicial-llt-solver.cpp
    src/decompositions/simplicial-ldlt-solver.cpp)

if(BUILD_WITH_CHOLMOD_SUPPORT)
  list(APPEND ${PROJECT_NAME}_DECOMPOSITIONS_SOURCES
       src/decompositions/cholmod.cpp)
endif(BUILD_WITH_CHOLMOD_SUPPORT)

if(BUILD_WITH_ACCELERATE_SUPPORT)
  list(APPEND ${PROJECT_NAME}_DECOMPOSITIONS_SOURCES
       src/decompositions/accelerate.cpp)
endif(BUILD_WITH_ACCELERATE_SUPPORT)

set(${PROJECT_NAME}_SOURCES
    ${${PROJECT_NAME}_SOLVERS_SOURCES}
    ${${PROJECT_NAME}_DECOMPOSITIONS_SOURCES}
    src/exception.cpp
    src/eigenpy.cpp
    src/numpy.cpp
    src/numpy-type.cpp
    src/matrix-float.cpp
    src/matrix-complex-float.cpp
    src/matrix-complex-double.cpp
    src/register.cpp
    src/matrix-double.cpp
    src/matrix-long-double.cpp
    src/matrix-complex-long-double.cpp
    src/matrix-bool.cpp
    src/matrix-char.cpp
    src/matrix-int8.cpp
    src/matrix-uint8.cpp
    src/matrix-int16.cpp
    src/matrix-uint16.cpp
    src/matrix-int32.cpp
    src/matrix-uint32.cpp
    src/matrix-windows-long.cpp
    src/matrix-windows-ulong.cpp
    src/matrix-mac-long.cpp
    src/matrix-mac-ulong.cpp
    src/matrix-int64.cpp
    src/matrix-uint64.cpp
    src/matrix-linux-long-long.cpp
    src/matrix-linux-ulong-long.cpp
    src/angle-axis.cpp
    src/quaternion.cpp
    src/geometry-conversion.cpp
    src/scipy-type.cpp
    src/std-vector.cpp
    src/optional.cpp
    src/type_info.cpp
    src/version.cpp)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES}
                                   ${${PROJECT_NAME}_HEADERS})

if(UNIX)
  get_relative_rpath(${CMAKE_INSTALL_LIBDIR} ${PROJECT_NAME}_INSTALL_RPATH)
  set_target_properties(
    ${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${${PROJECT_NAME}_INSTALL_RPATH}")
endif()
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
set_standard_output_directory(${PROJECT_NAME})
target_include_directories(
  ${PROJECT_NAME} SYSTEM
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

modernize_target_link_libraries(
  ${PROJECT_NAME}
  SCOPE
  PUBLIC
  TARGETS
  Eigen3::Eigen
  INCLUDE_DIRS
  ${EIGEN3_INCLUDE_DIR})

modernize_target_link_libraries(
  ${PROJECT_NAME}
  SCOPE
  PUBLIC
  TARGETS
  Python${PYTHON_VERSION_MAJOR}::NumPy
  INCLUDE_DIRS
  ${NUMPY_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIR})

# Links against CholMod
if(BUILD_WITH_CHOLMOD_SUPPORT)
  modernize_target_link_libraries(${PROJECT_NAME} SCOPE PUBLIC TARGETS
                                  CHOLMOD::CHOLMOD)
endif(BUILD_WITH_CHOLMOD_SUPPORT)

# Links against accelerate
if(BUILD_WITH_ACCELERATE_SUPPORT)
  # modernize_target_link_libraries(${PROJECT_NAME} SCOPE PUBLIC TARGETS
  # Accelerate)
  target_link_libraries(${PROJECT_NAME} PRIVATE "-framework accelerate")
endif(BUILD_WITH_ACCELERATE_SUPPORT)

if(SUFFIX_SO_VERSION)
  set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION})
endif(SUFFIX_SO_VERSION)

if(NOT WIN32)
  target_compile_options(
    ${PROJECT_NAME} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:-bigobj -MP>
                            "-Wno-conversion")
else()
  target_compile_options(${PROJECT_NAME}
                         PRIVATE $<$<CXX_COMPILER_ID:MSVC>:-bigobj -MP>)
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

# ----------------------------------------------------
# --- PYTHON LIBRARY ---------------------------------
# ----------------------------------------------------
add_subdirectory(python)

# ----------------------------------------------------
# --- UNIT TEST --------------------------------------
# ----------------------------------------------------
add_subdirectory(unittest)

pkg_config_append_libs(${PROJECT_NAME})
pkg_config_append_cflags("-I${PYTHON_INCLUDE_DIRS}")
pkg_config_append_cflags("-I${NUMPY_INCLUDE_DIRS}")

setup_project_finalize()
