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
#include "eigenpy/geometry.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
 
#include <boost/python.hpp>
namespace bp = boost::python;

Eigen::AngleAxisd testOutAngleAxis()
{
  return Eigen::AngleAxisd(.1,Eigen::Vector3d::UnitZ());
}

double testInAngleAxis(Eigen::AngleAxisd aa)
{
  return aa.angle();
}

Eigen::Quaterniond testOutQuaternion()
{
  Eigen::Quaterniond res(1,2,3,4);
  return res;
}
double testInQuaternion( Eigen::Quaterniond q )
{
  return q.norm(); 
}
double testInQuaternion_fx( eigenpy::Quaterniond_fx q )
{
  return q.norm(); 
}



BOOST_PYTHON_MODULE(geometry)
{
  eigenpy::enableEigenPy();

  eigenpy::exposeAngleAxis();
  eigenpy::exposeQuaternion();

  bp::def("testOutAngleAxis",&testOutAngleAxis);
  bp::def("testInAngleAxis",&testInAngleAxis);

  bp::def("testOutQuaternion",&testOutQuaternion);
  bp::def("testInQuaternion",&testInQuaternion);
  bp::def("testInQuaternion_fx",&testInQuaternion_fx);

}
 
