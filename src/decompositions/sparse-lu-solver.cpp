/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/decompositions/sparse/SparseLU.hpp"

namespace eigenpy {
void exposeSparseLUSolver() {
  using namespace Eigen;

  typedef SparseMatrix<double, ColMajor> ColMajorSparseMatrix;
  typedef typename ColMajorSparseMatrix::StorageIndex StorageIndex;
  typedef COLAMDOrdering<StorageIndex> Ordering;
  typedef SparseLU<ColMajorSparseMatrix, Ordering> SparseLUType;

  typedef typename SparseLUType::Scalar Scalar;
  typedef typename SparseLUType::SCMatrix SCMatrix;
  typedef Eigen::MappedSparseMatrix<Scalar, ColMajor, StorageIndex>
      MappedSparseMatrix;

  SparseLUMatrixLReturnTypeVisitor<SCMatrix>::expose(
      ("SparseLUMatrixLReturnType"));
  SparseLUMatrixUReturnTypeVisitor<SCMatrix, MappedSparseMatrix>::expose(
      ("SparseLUMatrixUReturnType"));
  SparseLUVisitor<ColMajorSparseMatrix>::expose("SparseLU");
}
}  // namespace eigenpy
