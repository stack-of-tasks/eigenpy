/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/solvers/IncompleteCholesky.hpp"

namespace eigenpy {
void exposeIncompleteCholesky() {
  using namespace Eigen;
  typedef SparseMatrix<double, ColMajor> ColMajorSparseMatrix;
  IncompleteCholeskyVisitor<ColMajorSparseMatrix>::expose("IncompleteCholesky");
}
}  // namespace eigenpy
