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

#if EIGEN_VERSION_AT_LEAST(3, 4, 90)
  typedef Eigen::Map<Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>>
      MappedSparseMatrix;
#else
  typedef Eigen::MappedSparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>
      MappedSparseMatrix;
#endif

  SparseLUMatrixLReturnTypeVisitor<SCMatrix>::expose(
      ("SparseLUMatrixLReturnType"));
  SparseLUMatrixUReturnTypeVisitor<SCMatrix, MappedSparseMatrix>::expose(
      ("SparseLUMatrixUReturnType"));
  SparseLUVisitor<ColMajorSparseMatrix>::expose("SparseLU");
}
}  // namespace eigenpy
