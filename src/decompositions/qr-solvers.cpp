/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/QR.hpp"

namespace eigenpy {
void exposeQRSolvers() {
  using namespace Eigen;
  HouseholderQRSolverVisitor<MatrixXd>::expose("HouseholderQR");
  FullPivHouseholderQRSolverVisitor<MatrixXd>::expose("FullPivHouseholderQR");
  ColPivHouseholderQRSolverVisitor<MatrixXd>::expose("ColPivHouseholderQR");
}
}  // namespace eigenpy
