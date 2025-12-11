# Add test that depends on the python wrapper of eigenpy library
# Those tests can import eigenpy
function(eigenpy_python_add_test name)
  set(options)
  set(oneValueArgs SCRIPT)
  set(multiValueArgs)
  cmake_parse_arguments(arg "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(arg_SCRIPT)
    set(script_path ${arg_SCRIPT})
  else()
    set(script_path ${CMAKE_CURRENT_SOURCE_DIR}/test_${name}.py)
  endif()

  set(test_name "Eigenpy: test_${name}.py")
  add_test(NAME ${test_name} COMMAND $<TARGET_FILE:Python::Interpreter> ${script_path})
  set_tests_properties(
    ${test_name}
    PROPERTIES ENVIRONMENT PYTHONPATH=$<TARGET_FILE_DIR:eigenpy_pywrap>/.. DEPENDS eigenpy_pywrap
  )
endfunction()

eigenpy_python_add_test(dimensions)
eigenpy_python_add_test(version)
eigenpy_python_add_test(EigenSolver)
eigenpy_python_add_test(GeneralizedEigenSolver)
eigenpy_python_add_test(GeneralizedSelfAdjointEigenSolver)
eigenpy_python_add_test(ComplexEigenSolver)
eigenpy_python_add_test(ComplexSchur)
eigenpy_python_add_test(SelfAdjointEigenSolver)
eigenpy_python_add_test(HessenbergDecomposition)

eigenpy_python_add_test(RealQZ)
eigenpy_python_add_test(RealSchur)
eigenpy_python_add_test(Tridiagonalization)
eigenpy_python_add_test(LLT)
eigenpy_python_add_test(LDLT)
eigenpy_python_add_test(FullPivLU)
eigenpy_python_add_test(PartialPivLU)
eigenpy_python_add_test(id)
eigenpy_python_add_test(QR)
eigenpy_python_add_test(BDCSVD)
eigenpy_python_add_test(JacobiSVD)

if(NOT WIN32)
  eigenpy_python_add_test(MINRES)
endif()

if(BUILD_TESTING_SCIPY)
  eigenpy_python_add_test(SimplicialLLT SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/decompositions/sparse/test_SimplicialLLT.py)
  eigenpy_python_add_test(SimplicialLDLT SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/decompositions/sparse/test_SimplicialLDLT.py)
  eigenpy_python_add_test(IncompleteCholesky SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/solvers/test_IncompleteCholesky.py)
  eigenpy_python_add_test(IncompleteLUT SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/solvers/test_IncompleteLUT.py)
  eigenpy_python_add_test(SparseLU SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/decompositions/sparse/test_SparseLU.py)
  eigenpy_python_add_test(SparseQR SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/decompositions/sparse/test_SparseQR.py)

  if(BUILD_WITH_CHOLMOD_SUPPORT)
    eigenpy_python_add_test(
      CholmodSimplicialLLT
      SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/decompositions/sparse/cholmod/test_CholmodSimplicialLLT.py
    )

    eigenpy_python_add_test(
      CholmodSimplicialLDLT
      SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/decompositions/sparse/cholmod/test_CholmodSimplicialLDLT.py
    )

    eigenpy_python_add_test(
      CholmodSupernodalLLT
      SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/decompositions/sparse/cholmod/test_CholmodSupernodalLLT.py
    )
  endif()

  if(BUILD_WITH_ACCELERATE_SUPPORT)
    eigenpy_python_add_test(Accelerate SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/decompositions/sparse/test_Accelerate.py)
  endif()
endif()
