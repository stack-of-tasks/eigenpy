/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/fwd.hpp"
#include "eigenpy/decompositions/decompositions.hpp"

#include "eigenpy/decompositions/sparse/accelerate/accelerate.hpp"

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

#define EXPOSE_ACCELERATE_DECOMPOSITION(name)                 \
  AccelerateImplVisitor<name<ColMajorSparseMatrix> >::expose( \
      EIGENPY_STRINGIZE(name))

  EXPOSE_ACCELERATE_DECOMPOSITION(AccelerateLLT);
  EXPOSE_ACCELERATE_DECOMPOSITION(AccelerateLDLT);
  EXPOSE_ACCELERATE_DECOMPOSITION(AccelerateLDLTUnpivoted);
  EXPOSE_ACCELERATE_DECOMPOSITION(AccelerateLDLTSBK);
  EXPOSE_ACCELERATE_DECOMPOSITION(AccelerateLDLTTPP);
  EXPOSE_ACCELERATE_DECOMPOSITION(AccelerateQR);
  EXPOSE_ACCELERATE_DECOMPOSITION(AccelerateCholeskyAtA);
}
}  // namespace eigenpy
