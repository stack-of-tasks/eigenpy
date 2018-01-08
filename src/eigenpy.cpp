/*
 * Copyright (c) 2015 LAAS-CNRS
 *
 * This file is part of eigenpy.
 * eigenpy is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 * eigenpy is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.  You should
 * have received a copy of the GNU Lesser General Public License along
 * with eigenpy.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "eigenpy/eigenpy.hpp"

namespace eigenpy
{

  /* Enable Eigen-Numpy serialization for a set of standard MatrixBase instance. */
  void enableEigenPy()
  {
    Exception::registerException();

    enableEigenPySpecific<Eigen::MatrixXd>();
    enableEigenPySpecific<Eigen::Matrix2d>();
    enableEigenPySpecific<Eigen::Matrix3d>();
    enableEigenPySpecific<Eigen::Matrix4d>();

    enableEigenPySpecific<Eigen::VectorXd>();
    enableEigenPySpecific<Eigen::Vector2d>();
    enableEigenPySpecific<Eigen::Vector3d>();
    enableEigenPySpecific<Eigen::Vector4d>();
  }

} // namespace eigenpy
