/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/LDLT.hpp"

namespace eigenpy {
void exposeLDLTSolver() {
  using namespace Eigen;
  LDLTSolverVisitor<MatrixXd>::expose("LDLT");
}
}  // namespace eigenpy
