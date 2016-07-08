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

#ifndef __eigenpy_fwd_hpp__
#define __eigenpy_fwd_hpp__

#include <boost/python.hpp>
#include <Eigen/Core>

namespace eigenpy
{
  template<typename D, typename Scalar = typename D::Scalar>
  struct UnalignedEquivalent
  {
    typedef Eigen::MatrixBase<D> MatType;
    typedef Eigen::Matrix<Scalar,
             D::RowsAtCompileTime,
             D::ColsAtCompileTime,
#ifndef EIGENPY_ALIGNED
             D::Options | Eigen::DontAlign,
#else
             D::Options,
#endif
             D::MaxRowsAtCompileTime,
             D::MaxColsAtCompileTime> type;
  };

} // namespace eigenpy

#endif // ifndef __eigenpy_fwd_hpp__
