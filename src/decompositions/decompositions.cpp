/*
 * Copyright 2020-2024 INRIA
 */

#include "eigenpy/decompositions/decompositions.hpp"

#include "eigenpy/fwd.hpp"

namespace eigenpy {

void exposeEigenSolver();
void exposeSelfAdjointEigenSolver();
void exposeLLTSolver();
void exposeLDLTSolver();
void exposeMINRESSolver();
void exposeSimplicialLLTSolver();
void exposeSimplicialLDLTSolver();
void exposePermutationMatrix();

void exposeDecompositions() {
  using namespace Eigen;

  exposeEigenSolver();
  exposeSelfAdjointEigenSolver();
  exposeLLTSolver();
  exposeLDLTSolver();
  exposeMINRESSolver();

  {
    bp::enum_<DecompositionOptions>("DecompositionOptions")
        .value("ComputeFullU", ComputeFullU)
        .value("ComputeThinU", ComputeThinU)
        .value("ComputeFullV", ComputeFullV)
        .value("ComputeThinV", ComputeThinV)
        .value("EigenvaluesOnly", EigenvaluesOnly)
        .value("ComputeEigenvectors", ComputeEigenvectors)
        .value("Ax_lBx", Ax_lBx)
        .value("ABx_lx", ABx_lx)
        .value("BAx_lx", BAx_lx);
  }

  // Expose sparse decompositions
  exposeSimplicialLLTSolver();
  exposeSimplicialLDLTSolver();

  exposePermutationMatrix();

#ifdef EIGENPY_WITH_CHOLMOD_SUPPORT
  exposeCholmod();
#endif

#ifdef EIGENPY_WITH_ACCELERATE_SUPPORT
  exposeAccelerate();
#endif
}
}  // namespace eigenpy
