/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/decompositions/BDCSVD.hpp"

namespace eigenpy {
void exposeBDCSVDSolver() {
  using namespace Eigen;
  BDCSVDVisitor<MatrixXd>::expose("BDCSVD");
}
}  // namespace eigenpy
