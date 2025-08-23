/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/solvers/MINRES.hpp"

namespace eigenpy {
void exposeMINRES() {
  using namespace Eigen;
  MINRESSolverVisitor<MatrixXd>::expose("MINRES");
}
}  // namespace eigenpy
