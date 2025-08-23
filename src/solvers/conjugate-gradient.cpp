/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/solvers/ConjugateGradient.hpp"

namespace eigenpy {
void exposeConjugateGradient() {
  using namespace Eigen;
  using Eigen::ConjugateGradient;
  using Eigen::IdentityPreconditioner;
  using Eigen::Lower;
  using IdentityConjugateGradient =
      ConjugateGradient<MatrixXd, Lower, IdentityPreconditioner>;

  ConjugateGradientVisitor<ConjugateGradient<MatrixXd, Lower>>::expose(
      "ConjugateGradient");
  ConjugateGradientVisitor<IdentityConjugateGradient>::expose(
      "IdentityConjugateGradient");
}
}  // namespace eigenpy
