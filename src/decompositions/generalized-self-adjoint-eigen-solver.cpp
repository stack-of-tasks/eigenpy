
/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/decompositions/GeneralizedSelfAdjointEigenSolver.hpp"

namespace eigenpy {
void exposeGeneralizedSelfAdjointEigenSolver() {
  using namespace Eigen;
  GeneralizedSelfAdjointEigenSolverVisitor<MatrixXd>::expose(
      "GeneralizedSelfAdjointEigenSolver");
}
}  // namespace eigenpy
