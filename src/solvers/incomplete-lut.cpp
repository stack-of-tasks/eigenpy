/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/solvers/IncompleteLUT.hpp"

namespace eigenpy {
void exposeIncompleteLUT() {
  typedef Eigen::SparseMatrix<double, Eigen::ColMajor> ColMajorSparseMatrix;
  IncompleteLUTVisitor<ColMajorSparseMatrix>::expose("IncompleteLUT");
}
}  // namespace eigenpy
