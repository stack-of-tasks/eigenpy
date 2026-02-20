# Copyright 2018-2026 Inria
# Copyright 2014-2019 CNRS

cmake_minimum_required(VERSION 3.22)

project(
  eigenpy
  VERSION 3.12.0
  DESCRIPTION "Bindings between Numpy and Eigen using Boost.Python"
  HOMEPAGE_URL "https://github.com/stack-of-tasks/eigenpy"
)

include(cmake/get-jrl-cmakemodules.cmake)

jrl_configure_defaults()

jrl_option(
  GENERATE_PYTHON_STUBS
  "Generate the Python stubs associated to the Python library"
  OFF
)

jrl_option(
  BUILD_WITH_CHOLMOD_SUPPORT
  "Build EigenPy with the Cholmod (LGPL) support. See CHOLMOD/Doc/License.txt for further details."
  OFF
)

jrl_option(BUILD_WITH_ACCELERATE_SUPPORT
           "Build EigenPy with the Accelerate support (Apple only)" OFF
)

jrl_option(BUILD_TESTING "Build the EigenPy unit tests" OFF)
jrl_option(BUILD_TESTING_SCIPY "Build the eigenpy unit tests with scipy" OFF)

if(BUILD_WITH_ACCELERATE_SUPPORT)
  if(NOT APPLE)
    message(WARNING "Accelerate support is only available on APPLE systems")
  endif()
endif()

# ----------------------------------------------------
# --- DEPENDENCIES -----------------------------------
# ----------------------------------------------------
jrl_find_package(Eigen3 CONFIG REQUIRED)

jrl_find_python(3.10 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
jrl_find_package(Boost CONFIG REQUIRED COMPONENTS python numpy)

if(BUILD_WITH_CHOLMOD_SUPPORT)
  jrl_find_package(CHOLMOD CONFIG REQUIRED)
endif()

if(BUILD_WITH_ACCELERATE_SUPPORT AND APPLE)
  jrl_find_package(Accelerate REQUIRED)
endif()

if(BUILD_TESTING AND BUILD_TESTING_SCIPY)
  jrl_check_python_module(scipy REQUIRED)
endif()

# ----------------------------------------------------
# --- INCLUDE ----------------------------------------
# ----------------------------------------------------
set(
  eigenpy_HEADERS
  # Utils
  include/eigenpy/utils/scalar-name.hpp
  include/eigenpy/utils/is-approx.hpp
  include/eigenpy/utils/is-aligned.hpp
  include/eigenpy/utils/traits.hpp
  include/eigenpy/utils/python-compat.hpp
  include/eigenpy/utils/empty-visitor.hpp
  # Solvers
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
  include/eigenpy/solvers/IncompleteLUT.hpp
  # Decompositions
  include/eigenpy/decompositions/sparse/SimplicialLLT.hpp
  include/eigenpy/decompositions/sparse/SimplicialLDLT.hpp
  include/eigenpy/decompositions/sparse/SparseLU.hpp
  include/eigenpy/decompositions/sparse/SparseQR.hpp
  include/eigenpy/decompositions/sparse/SimplicialCholesky.hpp
  include/eigenpy/decompositions/sparse/SparseSolverBase.hpp
  include/eigenpy/decompositions/sparse/LDLT.hpp
  include/eigenpy/decompositions/sparse/LLT.hpp
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
  include/eigenpy/decompositions/minres.hpp
  # Eigen
  include/eigenpy/eigen/EigenBase.hpp
  # Core
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
  include/eigenpy/version.hpp
)

if(BUILD_WITH_CHOLMOD_SUPPORT)
  list(
    APPEND eigenpy_HEADERS
    include/eigenpy/decompositions/sparse/cholmod/CholmodBase.hpp
    include/eigenpy/decompositions/sparse/cholmod/CholmodDecomposition.hpp
    include/eigenpy/decompositions/sparse/cholmod/CholmodSimplicialLDLT.hpp
    include/eigenpy/decompositions/sparse/cholmod/CholmodSimplicialLLT.hpp
    include/eigenpy/decompositions/sparse/cholmod/CholmodSupernodalLLT.hpp
  )
endif()

if(BUILD_WITH_ACCELERATE_SUPPORT)
  list(APPEND eigenpy_HEADERS include/eigenpy/decompositions/sparse/accelerate/Accelerate.hpp)
endif()

# ----------------------------------------------------
# --- TARGETS ----------------------------------------
# ----------------------------------------------------
set(
  eigenpy_SOURCES
  # Solvers
  src/solvers/preconditioners.cpp
  src/solvers/solvers.cpp
  src/solvers/minres.cpp
  src/solvers/bicgstab.cpp
  src/solvers/conjugate-gradient.cpp
  src/solvers/least-squares-conjugate-gradient.cpp
  src/solvers/incomplete-cholesky.cpp
  src/solvers/incomplete-lut.cpp
  # Decompositions
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
  src/decompositions/simplicial-ldlt-solver.cpp
  $<$<BOOL:${BUILD_WITH_CHOLMOD_SUPPORT}>:src/decompositions/cholmod.cpp>
  $<$<BOOL:${BUILD_WITH_ACCELERATE_SUPPORT}>:src/decompositions/accelerate.cpp>
  # Core
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
  src/version.cpp
)

add_library(eigenpy SHARED ${eigenpy_SOURCES} ${eigenpy_HEADERS})
add_library(eigenpy::eigenpy ALIAS eigenpy)
jrl_target_enforce_msvc_conformance(eigenpy PRIVATE)
target_compile_features(eigenpy PUBLIC cxx_std_17)

jrl_target_generate_config_header(eigenpy PUBLIC VERSION ${PROJECT_VERSION})
jrl_target_generate_warning_header(eigenpy PUBLIC)
jrl_target_generate_deprecated_header(eigenpy PUBLIC)

target_include_directories(
  eigenpy
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
  eigenpy
  PUBLIC
    Eigen3::Eigen
    Python::Module
    Python::NumPy
    Boost::python
    Boost::numpy
    $<$<BOOL:${BUILD_WITH_CHOLMOD_SUPPORT}>:SuiteSparse::CHOLMOD>
    $<$<BOOL:${BUILD_WITH_ACCELERATE_SUPPORT}>:Accelerate>
)

target_compile_definitions(
  eigenpy
  PUBLIC
    $<$<BOOL:${BUILD_WITH_CHOLMOD_SUPPORT}>:EIGENPY_WITH_CHOLMOD_SUPPORT>
    $<$<BOOL:${BUILD_WITH_ACCELERATE_SUPPORT}>:EIGENPY_WITH_ACCELERATE_SUPPORT>
)

if(SUFFIX_SO_VERSION)
  set_target_properties(eigenpy PROPERTIES SOVERSION ${PROJECT_VERSION})
endif()

if(CXX_COMPILER_ID STREQUAL "GNU" OR CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(eigenpy PRIVATE -Wno-conversion)
endif()

jrl_target_headers(eigenpy PUBLIC HEADERS ${eigenpy_HEADERS} BASE_DIRS include)

# ----------------------------------------------------
# --- EXPORT -----------------------------------------
# ----------------------------------------------------
jrl_add_export_component(TARGETS eigenpy NAME eigenpy)
jrl_export_package()

# ----------------------------------------------------
# --- PYTHON LIBRARY ---------------------------------
# ----------------------------------------------------
add_subdirectory(python)

# ----------------------------------------------------
# --- UNIT TEST --------------------------------------
# ----------------------------------------------------
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(unittest)
endif()

# ----------------------------------------------------
# --- ROS2 Support -----------------------------------
# ----------------------------------------------------
jrl_generate_ros2_package_files(
  INSTALL_CPP_PACKAGE_FILES [[NOT BUILD_STANDALONE_PYTHON_INTERFACE]]
  INSTALL_PYTHON_PACKAGE_FILES [[BUILD_PYTHON_INTERFACE]]
)

# ----------------------------------------------------
# --- SUMMARIES --------------------------------------
# ----------------------------------------------------
jrl_print_dependencies_summary()
jrl_print_options_summary()
