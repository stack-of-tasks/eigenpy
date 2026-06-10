/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/solvers/LeastSquaresConjugateGradient.hpp"

namespace eigenpy {
void exposeLeastSquaresConjugateGradient() {
  using namespace Eigen;
  using Eigen::LeastSquaresConjugateGradient;

  using Eigen::DiagonalPreconditioner;
  using Eigen::IdentityPreconditioner;
  using Eigen::LeastSquareDiagonalPreconditioner;

  using IdentityLeastSquaresConjugateGradient =
      LeastSquaresConjugateGradient<MatrixXd, IdentityPreconditioner>;
  using DiagonalLeastSquaresConjugateGradient = LeastSquaresConjugateGradient<
      MatrixXd, DiagonalPreconditioner<typename MatrixXd::Scalar>>;

  LeastSquaresConjugateGradientVisitor<LeastSquaresConjugateGradient<
      MatrixXd, LeastSquareDiagonalPreconditioner<MatrixXd::Scalar>>>::
      expose("LeastSquaresConjugateGradient");
  LeastSquaresConjugateGradientVisitor<IdentityLeastSquaresConjugateGradient>::
      expose("IdentityLeastSquaresConjugateGradient");
  LeastSquaresConjugateGradientVisitor<DiagonalLeastSquaresConjugateGradient>::
      expose("DiagonalLeastSquaresConjugateGradient");
}
}  // namespace eigenpy
