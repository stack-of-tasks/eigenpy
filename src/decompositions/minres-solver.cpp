/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/minres.hpp"

namespace eigenpy {
void exposeMINRESSolver() {
  using namespace Eigen;
  MINRESSolverVisitor<MatrixXd>::expose("MINRES");
}
}  // namespace eigenpy
