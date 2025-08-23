/*
 * Copyright 2017-2025 CNRS INRIA
 */

#include "eigenpy/solvers/solvers.hpp"

#include "eigenpy/fwd.hpp"

namespace eigenpy {

void exposeBiCGSTAB();
void exposeMINRES();
void exposeConjugateGradient();
void exposeLeastSquaresConjugateGradient();
void exposeIncompleteCholesky();
void exposeIncompleteLUT();

void exposeSolvers() {
  exposeBiCGSTAB();
  exposeMINRES();
  exposeConjugateGradient();
  exposeLeastSquaresConjugateGradient();
  exposeIncompleteCholesky();
  exposeIncompleteLUT();
}
}  // namespace eigenpy
