/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/LLT.hpp"

namespace eigenpy {
void exposeLLTSolver() {
  using namespace Eigen;
  LLTSolverVisitor<MatrixXd>::expose("LLT");
}
}  // namespace eigenpy
