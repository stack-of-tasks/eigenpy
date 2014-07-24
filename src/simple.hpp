/*
 * Copyright 2014, Nicolas Mansard, LAAS-CNRS
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

#ifndef __eigenpy_Simple_hpp__
#define __eigenpy_Simple_hpp__

#include <Eigen/Core>
#include <boost/python.hpp>

namespace eigenpy
{

  /* Enable Eigen-Numpy serialization for a set of standard MatrixBase instance. */
  void enableEigenPy();


  template<typename D>
  struct UnalignedEquivalent
  {
    typedef Eigen::MatrixBase<D> MatType;
    typedef Eigen::Matrix<typename D::Scalar,
			  D::RowsAtCompileTime,
			  D::ColsAtCompileTime,
			  D::Options | Eigen::DontAlign,
			  D::MaxRowsAtCompileTime,
			  D::MaxColsAtCompileTime>      type;
  };

  typedef UnalignedEquivalent<Eigen::MatrixXd> MatrixXd_fx;
  typedef UnalignedEquivalent<Eigen::Matrix3d> Matrix3d_fx;
  typedef UnalignedEquivalent<Eigen::Matrix4d> Matrix4d_fx;
  typedef UnalignedEquivalent<Eigen::VectorXd> VectorXd_fx;
  typedef UnalignedEquivalent<Eigen::Vector3d> Vector3d_fx;
  typedef UnalignedEquivalent<Eigen::Vector4d> Vector4d_fx;

} // namespace eigenpy

#endif // ifndef __eigenpy_Simple_hpp__
