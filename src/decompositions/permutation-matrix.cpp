/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/PermutationMatrix.hpp"

namespace eigenpy {
void exposePermutationMatrix() {
  using namespace Eigen;
  PermutationMatrixVisitor<Eigen::Dynamic>::expose("PermutationMatrix");
}
}  // namespace eigenpy
