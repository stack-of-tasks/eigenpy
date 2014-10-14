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

#ifndef __eigenpy_geometry_hpp__
#define __eigenpy_geometry_hpp__

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/quaternion.hpp"
#include "eigenpy/angle-axis.hpp"

namespace eigenpy
{
  typedef Eigen::Quaternion<double,Eigen::DontAlign> Quaterniond_fx;
  //typedef Eigen::AngleAxis<double> AngleAxis_fx;

  void exposeQuaternion();
  void exposeAngleAxis();

} // namespace eigenpy

#endif // define __eigenpy_geometry_hpp__
