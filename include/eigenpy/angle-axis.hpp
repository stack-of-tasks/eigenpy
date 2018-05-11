/*
 * Copyright 2014-2018, Nicolas Mansard, Justin Carpentier, LAAS-CNRS
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

#ifndef __eigenpy_angle_axis_hpp__
#define __eigenpy_angle_axis_hpp__

#include <boost/python.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "eigenpy/registration.hpp"

namespace eigenpy
{

  namespace bp = boost::python;

  template<typename AngleAxis>
  class AngleAxisVisitor
    :  public boost::python::def_visitor< AngleAxisVisitor<AngleAxis> >
  {

    typedef typename AngleAxis::Scalar Scalar;
    typedef typename AngleAxis::Vector3 Vector3;
    typedef typename AngleAxis::Matrix3 Matrix3;
    
    typedef typename Eigen::Quaternion<Scalar,0> Quaternion;
    
  public:

    template<class PyClass>
    void visit(PyClass& cl) const 
    {
      cl
      .def(bp::init<>("Default constructor"))
      .def(bp::init<Scalar,Vector3>
           ((bp::arg("angle"),bp::arg("axis")),
            "Initialize from angle and axis."))
      .def(bp::init<Matrix3>
           ((bp::arg("rotationMatrix")),
            "Initialize from a rotation matrix"))
      .def(bp::init<Quaternion>(bp::arg("quaternion"),"Initialize from a quaternion."))
      .def(bp::init<AngleAxis>(bp::arg("copy"),"Copy constructor."))
      
      /* --- Properties --- */
      .add_property("axis",
                    bp::make_function((const Vector3 & (AngleAxis::*)()const)&AngleAxis::axis,
                                      bp::return_value_policy<bp::copy_const_reference>()),
                    &AngleAxisVisitor::setAxis,"The rotation axis.")
      .add_property("angle",
                    (Scalar (AngleAxis::*)()const)&AngleAxis::angle,
                    &AngleAxisVisitor::setAngle,"The rotation angle.")
      
      /* --- Methods --- */
      .def("inverse",&AngleAxis::inverse,"Return the inverse rotation.")
      .def("fromRotationMatrix",&AngleAxis::template fromRotationMatrix<Matrix3>,
           bp::arg("Sets *this from a 3x3 rotation matrix."),bp::return_self<>())
      .def("toRotationMatrix",&AngleAxis::toRotationMatrix,"Constructs and returns an equivalent 3x3 rotation matrix.")
      .def("matrix",&AngleAxis::matrix,"Returns an equivalent rotation matrix.")
      .def("isApprox",(bool (AngleAxis::*)(const AngleAxis &))&AngleAxis::isApprox,
           "Returns true if *this is approximately equal to other.")
      .def("isApprox",(bool (AngleAxis::*)(const AngleAxis &, const Scalar prec))&AngleAxis::isApprox,
           bp::args("other","prec"),
           "Returns true if *this is approximately equal to other, within the precision determined by prec.")
      
      /* --- Operators --- */
      .def(bp::self * bp::other<Vector3>())
      .def(bp::self * bp::other<Quaternion>())
      .def(bp::self * bp::self)
      .def("__eq__",&AngleAxisVisitor::__eq__)
      .def("__ne__",&AngleAxisVisitor::__ne__)
      
      .def("__str__",&print)
      .def("__repr__",&print)
      ;
    }
    
  private:

    static void setAxis(AngleAxis & self, const Vector3 & axis)
    { self.axis() = axis; }

    static void setAngle( AngleAxis & self, const Scalar & angle)
    { self.angle() = angle; }

    static bool __eq__(const AngleAxis & u, const AngleAxis & v)
    { return u.isApprox(v); }
    static bool __ne__(const AngleAxis & u, const AngleAxis & v)
    { return !__eq__(u,v); }

    static std::string print(const AngleAxis & self)
    {
      std::stringstream ss;
      ss << "angle: " << self.angle() << std::endl;
      ss << "axis: " << self.axis().transpose() << std::endl;
      
      return ss.str();
    }

  public:

    static void expose()
    {
      if(check_registration<AngleAxis>()) return;
      
      bp::class_<AngleAxis>("AngleAxis",
                            "AngleAxis representation of rotations.\n\n",
                            bp::no_init)
      .def(AngleAxisVisitor<AngleAxis>());
    }

  };

} // namespace eigenpy

#endif //ifndef __eigenpy_angle_axis_hpp__
