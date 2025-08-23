/*
 * Copyright 2020-2024 INRIA
 */

#include "eigenpy/decompositions/decompositions.hpp"
#include "eigenpy/solvers/MINRES.hpp"

#include "eigenpy/fwd.hpp"

namespace eigenpy {

void exposeEigenSolver();
void exposeGeneralizedEigenSolver();
void exposeSelfAdjointEigenSolver();
void exposeGeneralizedSelfAdjointEigenSolver();
void exposeHessenbergDecomposition();
void exposeRealQZ();
void exposeRealSchur();
void exposeTridiagonalization();
void exposeComplexEigenSolver();
void exposeComplexSchur();
void exposeLLTSolver();
void exposeLDLTSolver();
void exposeFullPivLUSolver();
void exposePartialPivLUSolver();
void exposeQRSolvers();
void exposeSimplicialLLTSolver();
void exposeSimplicialLDLTSolver();
void exposeSparseLUSolver();
void exposeSparseQRSolver();
void exposePermutationMatrix();
void exposeBDCSVDSolver();
void exposeJacobiSVDSolver();

void exposeBackwardCompatibilityAliases() {
  typedef Eigen::MatrixXd MatrixXd;
  MINRESSolverVisitor<MatrixXd>::expose("MINRES");
}

void exposeDecompositions() {
  using namespace Eigen;

  exposeEigenSolver();
  exposeGeneralizedEigenSolver();
  exposeSelfAdjointEigenSolver();
  exposeGeneralizedSelfAdjointEigenSolver();
  exposeHessenbergDecomposition();
  exposeRealQZ();
  exposeRealSchur();
  exposeTridiagonalization();
  exposeComplexEigenSolver();
  exposeComplexSchur();
  exposeLLTSolver();
  exposeLDLTSolver();
  exposeFullPivLUSolver();
  exposePartialPivLUSolver();
  exposeQRSolvers();
  exposeBDCSVDSolver();
  exposeJacobiSVDSolver();

  {
    bp::enum_<DecompositionOptions>("DecompositionOptions")
        .value("ComputeFullU", ComputeFullU)
        .value("ComputeThinU", ComputeThinU)
        .value("ComputeFullV", ComputeFullV)
        .value("ComputeThinV", ComputeThinV)
        .value("EigenvaluesOnly", EigenvaluesOnly)
        .value("ComputeEigenvectors", ComputeEigenvectors)
        .value("Ax_lBx", Ax_lBx)
        .value("ABx_lx", ABx_lx)
        .value("BAx_lx", BAx_lx);
  }

  // Expose sparse decompositions
  exposeSimplicialLLTSolver();
  exposeSimplicialLDLTSolver();
  exposeSparseLUSolver();
  exposeSparseQRSolver();

  exposePermutationMatrix();

#ifdef EIGENPY_WITH_CHOLMOD_SUPPORT
  exposeCholmod();
#endif

#ifdef EIGENPY_WITH_ACCELERATE_SUPPORT
  exposeAccelerate();
#endif

  exposeBackwardCompatibilityAliases();
}
}  // namespace eigenpy
