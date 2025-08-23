/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/decompositions/PartialPivLU.hpp"

namespace eigenpy {
void exposePartialPivLUSolver() {
  using namespace Eigen;
  PartialPivLUSolverVisitor<MatrixXd>::expose("PartialPivLU");
}
}  // namespace eigenpy
