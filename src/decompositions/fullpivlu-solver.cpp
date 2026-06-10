/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/decompositions/FullPivLU.hpp"

namespace eigenpy {
void exposeFullPivLUSolver() {
  using namespace Eigen;
  FullPivLUSolverVisitor<MatrixXd>::expose("FullPivLU");
}
}  // namespace eigenpy
