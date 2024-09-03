/*
 * Copyright 2024 INRIA
 */

#include <iostream>

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/type_info.hpp"

struct Dummy {};

BOOST_PYTHON_MODULE(type_info) {
  using namespace Eigen;
  namespace bp = boost::python;
  eigenpy::enableEigenPy();

  eigenpy::expose_boost_type_info<int>();
  eigenpy::expose_boost_type_info<std::string>();

  bp::class_<Dummy>("Dummy").def(eigenpy::TypeInfoVisitor<Dummy>());
}
