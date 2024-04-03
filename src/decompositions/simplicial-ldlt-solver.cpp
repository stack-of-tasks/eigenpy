/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/sparse/LDLT.hpp"

namespace eigenpy {
void exposeSimplicialLDLTSolver() {
  using namespace Eigen;
  typedef SparseMatrix<double, ColMajor> ColMajorSparseMatrix;
  SimplicialLDLTVisitor<ColMajorSparseMatrix>::expose("SimplicialLDLT");
}
}  // namespace eigenpy
