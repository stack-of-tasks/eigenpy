/*
 * Copyright 2017 CNRS
 */

#include <Eigen/Core>

#if EIGEN_VERSION_AT_LEAST(3, 2, 0)
#include "eigenpy/solvers/BasicPreconditioners.hpp"
#include "eigenpy/solvers/preconditioners.hpp"
// #include "eigenpy/solvers/BFGSPreconditioners.hpp"

namespace eigenpy {

void exposePreconditioners() {
  using namespace Eigen;

  DiagonalPreconditionerVisitor<double>::expose();
#if EIGEN_VERSION_AT_LEAST(3, 3, 5)
  LeastSquareDiagonalPreconditionerVisitor<double>::expose();
#endif
  IdentityPreconditionerVisitor::expose();
  //      LimitedBFGSPreconditionerBaseVisitor<
  //      LimitedBFGSPreconditioner<double,Eigen::Dynamic,Eigen::Upper|Eigen::Lower>
  //      >::expose("LimitedBFGSPreconditioner");
}

}  // namespace eigenpy

#endif
