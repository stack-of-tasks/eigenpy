
/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/decompositions/Tridiagonalization.hpp"

namespace eigenpy {
void exposeTridiagonalization() {
  using namespace Eigen;
  TridiagonalizationVisitor<MatrixXd>::expose("Tridiagonalization");
}
}  // namespace eigenpy
