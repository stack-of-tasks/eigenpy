/*
 * Copyright 2020 INRIA
 */

#include <boost/python.hpp>
#include <Eigen/Core>

#include "eigenpy/decompositions/EigenSolver.hpp"

namespace eigenpy
{
  void exposeDecompositions()
  {
    using namespace Eigen;
    namespace bp = boost::python;
    
    EigenSolverVisitor<Eigen::MatrixXd>::expose("EigenSolver");
    
  }
} // namespace eigenpy



