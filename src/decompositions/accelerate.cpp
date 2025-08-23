/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/fwd.hpp"
#include "eigenpy/decompositions/decompositions.hpp"

#include "eigenpy/decompositions/sparse/accelerate/Accelerate.hpp"

namespace eigenpy {

void exposeAccelerate() {
  using namespace Eigen;

  typedef Eigen::SparseMatrix<double, Eigen::ColMajor> ColMajorSparseMatrix;
  //  typedef Eigen::SparseMatrix<double,Eigen::RowMajor> RowMajorSparseMatrix;

  bp::enum_<SparseOrder_t>("SparseOrder")
      .value("SparseOrderUser", SparseOrderUser)
      .value("SparseOrderAMD", SparseOrderAMD)
      .value("SparseOrderMetis", SparseOrderMetis)
      .value("SparseOrderCOLAMD", SparseOrderCOLAMD);

#define EXPOSE_ACCELERATE_DECOMPOSITION(name, doc)           \
  AccelerateImplVisitor<name<ColMajorSparseMatrix>>::expose( \
      EIGENPY_STRINGIZE(name), doc)

  EXPOSE_ACCELERATE_DECOMPOSITION(
      AccelerateLLT,
      "A direct Cholesky (LLT) factorization and solver based on Accelerate.");
  EXPOSE_ACCELERATE_DECOMPOSITION(AccelerateLDLT,
                                  "The default Cholesky (LDLT) factorization "
                                  "and solver based on Accelerate.");
  EXPOSE_ACCELERATE_DECOMPOSITION(
      AccelerateLDLTUnpivoted,
      "A direct Cholesky-like LDL^T factorization and solver based on "
      "Accelerate with only 1x1 pivots and no pivoting.");
  EXPOSE_ACCELERATE_DECOMPOSITION(
      AccelerateLDLTSBK,
      "A direct Cholesky (LDLT) factorization and solver based on Accelerate "
      "with Supernode Bunch-Kaufman and static pivoting.");
  EXPOSE_ACCELERATE_DECOMPOSITION(
      AccelerateLDLTTPP,
      "A direct Cholesky (LDLT) factorization and solver based on Accelerate "
      "with full threshold partial pivoting.");
  EXPOSE_ACCELERATE_DECOMPOSITION(
      AccelerateQR, "A QR factorization and solver based on Accelerate.");
  EXPOSE_ACCELERATE_DECOMPOSITION(
      AccelerateCholeskyAtA,
      "A QR factorization and solver based on Accelerate without storing Q "
      "(equivalent to A^TA = R^T R).");
}
}  // namespace eigenpy
