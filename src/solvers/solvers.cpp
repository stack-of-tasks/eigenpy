/*
 * Copyright 2017-2020 CNRS INRIA
 */

#include <Eigen/Core>

#if EIGEN_VERSION_AT_LEAST(3, 2, 0)

#include "eigenpy/solvers/ConjugateGradient.hpp"
#include "eigenpy/solvers/solvers.hpp"

#if EIGEN_VERSION_AT_LEAST(3, 3, 5)
#include "eigenpy/solvers/LeastSquaresConjugateGradient.hpp"
#endif

namespace eigenpy {
void exposeSolvers() {
  using namespace Eigen;
  ConjugateGradientVisitor<
      ConjugateGradient<MatrixXd, Lower | Upper> >::expose();
#if EIGEN_VERSION_AT_LEAST(3, 3, 5)
  LeastSquaresConjugateGradientVisitor<LeastSquaresConjugateGradient<
      MatrixXd,
      LeastSquareDiagonalPreconditioner<MatrixXd::Scalar> > >::expose();
#endif

  // Conjugate gradient with limited BFGS preconditioner
  ConjugateGradientVisitor<
      ConjugateGradient<MatrixXd, Lower | Upper, IdentityPreconditioner> >::
      expose("IdentityConjugateGradient");
  //    ConjugateGradientVisitor<
  //    ConjugateGradient<MatrixXd,Lower|Upper,LimitedBFGSPreconditioner<double,Dynamic,Lower|Upper>
  //    > >::expose("LimitedBFGSConjugateGradient");
}
}  // namespace eigenpy

#endif
