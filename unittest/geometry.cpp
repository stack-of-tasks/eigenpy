/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2023, INRIA
 */

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/geometry.hpp"

namespace bp = boost::python;

Eigen::AngleAxisd testOutAngleAxis() {
  return Eigen::AngleAxisd(.1, Eigen::Vector3d::UnitZ());
}

double testInAngleAxis(Eigen::AngleAxisd aa) { return aa.angle(); }

Eigen::Quaterniond testOutQuaternion() {
  Eigen::Quaterniond res(1, 2, 3, 4);
  return res;
}
double testInQuaternion(Eigen::Quaterniond q) { return q.norm(); }

BOOST_PYTHON_MODULE(geometry) {
  eigenpy::enableEigenPy();

  eigenpy::exposeAngleAxis();
  eigenpy::exposeQuaternion();

  bp::def("testOutAngleAxis", &testOutAngleAxis);
  bp::def("testInAngleAxis", &testInAngleAxis);

  bp::def("testOutQuaternion", &testOutQuaternion);
  bp::def("testInQuaternion", &testInQuaternion);
}
