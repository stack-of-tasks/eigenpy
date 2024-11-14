
/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/SelfAdjointEigenSolver.hpp"

namespace eigenpy {
void exposeSelfAdjointEigenSolver() {
  using namespace Eigen;
  SelfAdjointEigenSolverVisitor<MatrixXd>::expose("SelfAdjointEigenSolver");
}
}  // namespace eigenpy
