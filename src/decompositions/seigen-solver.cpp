
/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/EigenSolver.hpp"

namespace eigenpy {
void exposeEigenSolver() {
  using namespace Eigen;
  EigenSolverVisitor<MatrixXd>::expose("EigenSolver");
}
}  // namespace eigenpy
