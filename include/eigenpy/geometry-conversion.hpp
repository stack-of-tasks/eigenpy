/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2023, INRIA
 */

#ifndef __eigenpy_geometry_conversion_hpp__
#define __eigenpy_geometry_conversion_hpp__

#include "eigenpy/fwd.hpp"

namespace eigenpy {

template <typename Scalar, int Options = 0>
struct EulerAnglesConvertor {
  typedef typename Eigen::Matrix<Scalar, 3, 1, Options> Vector3;
  typedef typename Eigen::Matrix<Scalar, 3, 3, Options> Matrix3;
  typedef typename Vector3::Index Index;

  typedef typename Eigen::AngleAxis<Scalar> AngleAxis;

  static void expose() {
    bp::def("toEulerAngles", &EulerAnglesConvertor::toEulerAngles,
            bp::args("rotation_matrix", "a0", "a1", "a2"),
            "It returns the Euler-angles of the rotation matrix mat using the "
            "convention defined by the triplet (a0,a1,a2).");

    bp::def("fromEulerAngles", &EulerAnglesConvertor::fromEulerAngles,
            bp::args("euler_angles", "a0", "a1", "a2"),
            "It returns the rotation matrix associated to the Euler angles "
            "using the convention defined by the triplet (a0,a1,a2).");
  }

  static Vector3 toEulerAngles(const Matrix3& mat, Index a0, Index a1,
                               Index a2) {
    return mat.eulerAngles(a0, a1, a2);
  }

  static Matrix3 fromEulerAngles(const Vector3& ea, Index a0, Index a1,
                                 Index a2) {
    Matrix3 mat;
    mat = AngleAxis(ea[0], Vector3::Unit(a0)) *
          AngleAxis(ea[1], Vector3::Unit(a1)) *
          AngleAxis(ea[2], Vector3::Unit(a2));
    return mat;
  }
};

}  // namespace eigenpy

#endif  // define __eigenpy_geometry_conversion_hpp__
