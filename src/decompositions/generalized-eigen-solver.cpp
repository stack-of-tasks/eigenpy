
/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/decompositions/GeneralizedEigenSolver.hpp"

namespace eigenpy {
void exposeGeneralizedEigenSolver() {
  using namespace Eigen;
  GeneralizedEigenSolverVisitor<MatrixXd>::expose("GeneralizedEigenSolver");
}
}  // namespace eigenpy
