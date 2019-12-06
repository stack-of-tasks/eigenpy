/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/version.hpp"
#include "eigenpy/geometry.hpp"
#include "eigenpy/solvers/solvers.hpp"
#include "eigenpy/solvers/preconditioners.hpp"

#include <iostream>
#include <boost/python/scope.hpp>

using namespace eigenpy;

BOOST_PYTHON_MODULE(eigenpy)
{
  enableEigenPy();
  
  bp::scope().attr("__version__") = eigenpy::printVersion();
  bp::scope().attr("__raw_version__") = bp::str(EIGENPY_VERSION);
  bp::def("checkVersionAtLeast",&eigenpy::checkVersionAtLeast,
          bp::args("major_version","minor_version","patch_version"),
          "Checks if the current version of EigenPy is at least the version provided by the input arguments.");
  
  exposeAngleAxis();
  exposeQuaternion();
  exposeGeometryConversion();
  
  {
    boost::python::scope solvers = boost::python::class_<SolversScope>("solvers");
    exposeSolvers();
    exposePreconditioners();
  }
  
}
