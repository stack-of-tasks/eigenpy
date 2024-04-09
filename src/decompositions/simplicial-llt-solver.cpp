/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/sparse/LLT.hpp"

namespace eigenpy {
void exposeSimplicialLLTSolver() {
  using namespace Eigen;
  typedef SparseMatrix<double, ColMajor> ColMajorSparseMatrix;
  SimplicialLLTVisitor<ColMajorSparseMatrix>::expose("SimplicialLLT");
}
}  // namespace eigenpy
