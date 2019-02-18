/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/geometry.hpp"
#include "eigenpy/solvers/solvers.hpp"
#include "eigenpy/solvers/preconditioners.hpp"

#include <iostream>
#include <boost/python/scope.hpp>

using namespace eigenpy;

BOOST_PYTHON_MODULE(eigenpy)
{
  enableEigenPy();
  exposeAngleAxis();
  exposeQuaternion();
  exposeGeometryConversion();
  
  {
    boost::python::scope solvers = boost::python::class_<SolversScope>("solvers");
    exposeSolvers();
    exposePreconditioners();
  }
  
}
