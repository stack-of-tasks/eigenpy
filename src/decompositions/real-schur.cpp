
/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/decompositions/RealSchur.hpp"

namespace eigenpy {
void exposeRealSchur() {
  using namespace Eigen;
  RealSchurVisitor<MatrixXd>::expose("RealSchur");
}
}  // namespace eigenpy
