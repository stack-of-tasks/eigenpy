/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/solvers/BiCGSTAB.hpp"

namespace eigenpy {
void exposeBiCGSTAB() {
  using namespace Eigen;
  using Eigen::BiCGSTAB;
  using Eigen::IdentityPreconditioner;
  using IdentityBiCGSTAB = BiCGSTAB<MatrixXd, IdentityPreconditioner>;

  BiCGSTABVisitor<BiCGSTAB<MatrixXd>>::expose("BiCGSTAB");
  BiCGSTABVisitor<IdentityBiCGSTAB>::expose("IdentityBiCGSTAB");
}
}  // namespace eigenpy
