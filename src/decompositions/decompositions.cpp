/*
 * Copyright 2020-2021 INRIA
 */

#include "eigenpy/decompositions/decompositions.hpp"

#include "eigenpy/decompositions/EigenSolver.hpp"
#include "eigenpy/decompositions/LDLT.hpp"
#include "eigenpy/decompositions/LLT.hpp"
#include "eigenpy/decompositions/SelfAdjointEigenSolver.hpp"
#include "eigenpy/decompositions/minres.hpp"
#include "eigenpy/fwd.hpp"

namespace eigenpy {
void exposeDecompositions() {
  using namespace Eigen;

  EigenSolverVisitor<MatrixXd>::expose("EigenSolver");
  SelfAdjointEigenSolverVisitor<MatrixXd>::expose("SelfAdjointEigenSolver");
  LLTSolverVisitor<MatrixXd>::expose("LLT");
  LDLTSolverVisitor<MatrixXd>::expose("LDLT");

  MINRESSolverVisitor<MatrixXd>::expose("MINRES");

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
}
}  // namespace eigenpy
