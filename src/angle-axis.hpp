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

#ifndef __eigenpy_angle_axis_hpp__
#define __eigenpy_angle_axis_hpp__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/python.hpp>
#include "eigenpy/eigenpy.hpp"

namespace eigenpy
{
  template<>
  struct UnalignedEquivalent<Eigen::AngleAxisd>
  {
    typedef Eigen::AngleAxis<double> type;
  };

  namespace bp = boost::python;

  template<typename D>
  class AngleAxisVisitor
    :  public boost::python::def_visitor< AngleAxisVisitor<D> >
  {
    typedef D AngleAxis;
    typedef typename eigenpy::UnalignedEquivalent<D>::type AngleAxisUnaligned;

    typedef typename AngleAxisUnaligned::Scalar Scalar;
    typedef typename eigenpy::UnalignedEquivalent<typename AngleAxisUnaligned::Vector3>::type Vector3;
    typedef typename eigenpy::UnalignedEquivalent<typename AngleAxisUnaligned::Matrix3>::type Matrix3;

    typedef eigenpy::UnalignedEquivalent<Eigen::Quaternion<Scalar> > Quaternion;

  public:

    template<class PyClass>
    void visit(PyClass& cl) const 
    {
      cl
	.def(bp::init<Scalar,Vector3>
	     ((bp::arg("angle"),bp::arg("axis")),
	      "Initialize from angle and axis"))
	.def(bp::init<Matrix3>
	     ((bp::arg("rotationMatrix")),
	      "Initialize from a rotation matrix"))
	.def("__init__",bp::make_constructor(&AngleAxisVisitor::constructFromQuaternion,
					     bp::default_call_policies(),
					     (bp::arg("quaternion"))),"Initialize from quaternion")
	.def(bp::init<AngleAxisUnaligned>((bp::arg("clone"))))

	.def("matrix",&AngleAxisUnaligned::toRotationMatrix,"Return the corresponding rotation matrix 3x3.")
	.def("vector",&AngleAxisVisitor::toVector3,"Return the correspond angle*axis vector3")
	.add_property("axis",&AngleAxisVisitor::getAxis,&AngleAxisVisitor::setAxis)
	.add_property("angle",&AngleAxisVisitor::getAngle,&AngleAxisVisitor::setAngle)

	/* --- Methods --- */
	.def("normalize",&AngleAxisVisitor::normalize,"Normalize the axis vector (without changing the angle).")
	.def("inverse",&AngleAxisUnaligned::inverse,"Return the inverse rotation.")
	.def("apply",&AngleAxisVisitor::apply,(bp::arg("vector3")),"Apply the rotation map to the vector")

	/* --- Operators --- */
	.def(bp::self * bp::other<Vector3>())
	.def("__eq__",&AngleAxisVisitor::__eq__)
	.def("__ne__",&AngleAxisVisitor::__ne__)
	.def("__abs__",&AngleAxisVisitor::getAngleAbs)
	;
    }
  private:
  
    static AngleAxisUnaligned* constructFromQuaternion(const Eigen::Quaternion<Scalar,Eigen::DontAlign> & qu)
    {
      Eigen::Quaternion<Scalar> q = qu;
      return new AngleAxisUnaligned(q);
    }
 
    static Vector3 apply(const AngleAxisUnaligned & r, const Vector3 & v ) { return r*v; }

    static Vector3 getAxis(const AngleAxisUnaligned& self) { return self.axis(); }
    static void setAxis(AngleAxisUnaligned& self, const Vector3 & r)
    {
      self = AngleAxisUnaligned( self.angle(),r );
    }

    static double getAngle(const AngleAxisUnaligned& self) { return self.angle(); }
    static void setAngle( AngleAxisUnaligned& self, const double & th) 
    {
      self = AngleAxisUnaligned( th,self.axis() );
    }
    static double getAngleAbs(const AngleAxisUnaligned& self) { return std::abs(self.angle()); }

    static bool __eq__(const AngleAxisUnaligned & u, const AngleAxisUnaligned & v)
    {
      return u.isApprox(v);
    }
    static bool __ne__(const AngleAxisUnaligned & u, const AngleAxisUnaligned & v)
    {
      return !__eq__(u,v); 
    }

    static Vector3 toVector3( const AngleAxisUnaligned & self ) { return self.axis()*self.angle(); }
    static void normalize( AngleAxisUnaligned & self )
    {
      setAxis(self,self.axis() / self.axis().norm());
    }

  private:
  
    static PyObject* convert(AngleAxis const& q)
    {
      AngleAxisUnaligned qx = q;
      return boost::python::incref(boost::python::object(qx).ptr());
    }

  public:

    static void expose()
    {
      bp::class_<AngleAxisUnaligned>("AngleAxis",
				     "AngleAxis representation of rotations.\n\n",
				     bp::init<>())
	.def(AngleAxisVisitor<D>());
    
      // TODO: check the problem of fix-size Angle Axis.
      //bp::to_python_converter< AngleAxis,AngleAxisVisitor<D> >();

    }

  };

} // namespace eigenpy

#endif //ifndef __eigenpy_angle_axis_hpp__
