/*
 * Copyright 2020 INRIA
 */

#include <boost/python.hpp>
#include <Eigen/Core>

#include "eigenpy/decompositions/EigenSolver.hpp"
#include "eigenpy/decompositions/SelfAdjointEigenSolver.hpp"

namespace eigenpy
{
  void exposeDecompositions()
  {
    using namespace Eigen;
    namespace bp = boost::python;
    
    EigenSolverVisitor<Eigen::MatrixXd>::expose("EigenSolver");
    SelfAdjointEigenSolverVisitor<Eigen::MatrixXd>::expose("SelfAdjointEigenSolver");

    {
      using namespace Eigen;
      bp::enum_<DecompositionOptions>("DecompositionOptions")
      .value("ComputeFullU",ComputeFullU)
      .value("ComputeThinU",ComputeThinU)
      .value("ComputeFullV",ComputeFullV)
      .value("ComputeThinV",ComputeThinV)
      .value("EigenvaluesOnly",EigenvaluesOnly)
      .value("ComputeEigenvectors",ComputeEigenvectors)
      .value("Ax_lBx",Ax_lBx)
      .value("ABx_lx",ABx_lx)
      .value("BAx_lx",BAx_lx)
      ;
    }
    
  }
} // namespace eigenpy
