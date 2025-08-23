/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/decompositions/sparse/SparseQR.hpp"

namespace eigenpy {
void exposeSparseQRSolver() {
  using namespace Eigen;

  typedef SparseMatrix<double, ColMajor> ColMajorSparseMatrix;
  typedef typename ColMajorSparseMatrix::StorageIndex StorageIndex;
  typedef COLAMDOrdering<StorageIndex> Ordering;
  typedef SparseQR<ColMajorSparseMatrix, Ordering> SparseQRType;

  SparseQRMatrixQTransposeReturnTypeVisitor<SparseQRType>::expose(
      "SparseQRMatrixQTransposeReturnType");
  SparseQRMatrixQReturnTypeVisitor<SparseQRType>::expose(
      "SparseQRMatrixQReturnType");
  SparseQRVisitor<SparseQRType>::expose("SparseQR");
}
}  // namespace eigenpy
