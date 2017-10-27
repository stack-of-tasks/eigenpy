//
// Copyright (c) 2017, Justin Carpentier, CNRS
//
// This file is part of eigenpy
// eigenpy is free software: you can redistribute it
// and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version.
//
// eigenpy is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Lesser Public License for more details. You should have
// received a copy of the GNU Lesser General Public License along with
// eigenpy If not, see
// <http://www.gnu.org/licenses/>.

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/geometry.hpp"
#include "eigenpy/solvers/solvers.hpp"
#include "eigenpy/solvers/preconditioners.hpp"

#include <iostream>
#include <boost/python/scope.hpp>

namespace bp = boost::python;
using namespace eigenpy;


BOOST_PYTHON_MODULE(eigenpy)
{
  enableEigenPy();
  exposeAngleAxis();
  exposeQuaternion();
  
  {
    bp::scope solvers = bp::class_<SolversScope>("solvers");
    exposeSolvers();
    exposePreconditioners();
  }
  
}
