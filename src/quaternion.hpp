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

#ifndef __eigenpy_quaternion_hpp__
#define __eigenpy_quaternion_hpp__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "eigenpy/eigenpy.hpp"
#include "eigenpy/exception.hpp"

namespace eigenpy
{

  class ExceptionIndex : public Exception
  {
  public:
    int index;
    ExceptionIndex(int index,int imin,int imax) : Exception("")
    {
      std::ostringstream oss; oss << "Index " << index << " out of range " << imin << ".."<< imax <<".";
      message = oss.str();
    }
  };

  template<>
  struct UnalignedEquivalent<Eigen::Quaterniond>
  {
#ifndef EIGENPY_ALIGNED
    typedef Eigen::Quaternion<Eigen::Quaterniond::Scalar,Eigen::DontAlign> type;
#else
    typedef Eigen::Quaterniond type;
#endif
  };

  namespace bp = boost::python;

  template<typename Quaternion>
  class QuaternionVisitor
    :  public boost::python::def_visitor< QuaternionVisitor<Quaternion> >
  {
    typedef Eigen::QuaternionBase<Quaternion> QuaternionBase;
    typedef typename eigenpy::UnalignedEquivalent<Quaternion>::type QuaternionUnaligned;

    typedef typename QuaternionBase::Scalar Scalar;
    typedef typename QuaternionUnaligned::Coefficients Coefficients;
#ifndef EIGENPY_ALIGNED
    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vector3;
    typedef Eigen::Matrix<Scalar,3,3,Eigen::DontAlign> Matrix3;
#else
    typedef Eigen::Matrix<Scalar,3,1,0> Vector3;
    typedef Eigen::Matrix<Scalar,3,3,0> Matrix3;
#endif

    typedef Eigen::AngleAxis<Scalar> AngleAxisUnaligned;

  public:

    /* Conversion from C++ to Python. */
    static PyObject* convert(Quaternion const& q)
    {
      QuaternionUnaligned qx = q;
      return boost::python::incref(boost::python::object(qx).ptr());
    }

    template<class PyClass>
    void visit(PyClass& cl) const 
    {
      cl
	.def(bp::init<Matrix3>((bp::arg("matrixRotation")),"Initialize from rotation matrix."))
	.def(bp::init<AngleAxisUnaligned>((bp::arg("angleaxis")),"Initialize from angle axis."))
	.def(bp::init<QuaternionUnaligned>((bp::arg("clone")),"Copy constructor."))
	.def("__init__",bp::make_constructor(&QuaternionVisitor::fromTwoVectors,
					     bp::default_call_policies(),
					     (bp::arg("u"),bp::arg("v"))),"Initialize from two vector u,v")
	.def(bp::init<Scalar,Scalar,Scalar,Scalar>
	     ((bp::arg("w"),bp::arg("x"),bp::arg("y"),bp::arg("z")),
	      "Initialize from coefficients.\n\n"
	      "... note:: The order of coefficients is *w*, *x*, *y*, *z*. "
	      "The [] operator numbers them differently, 0...4 for *x* *y* *z* *w*!"))

	/* --- Methods --- */
	.def("coeffs",&QuaternionVisitor::coeffs)
	.def("matrix",&QuaternionUnaligned::toRotationMatrix)
      
	.def("setFromTwoVectors",&QuaternionVisitor::setFromTwoVectors,((bp::arg("u"),bp::arg("v"))))
	.def("conjugate",&QuaternionUnaligned::conjugate)
	.def("inverse",&QuaternionUnaligned::inverse)
	.def("norm",&QuaternionUnaligned::norm)
	.def("normalize",&QuaternionUnaligned::normalize)
	.def("normalized",&QuaternionUnaligned::normalized)
	.def("apply",&QuaternionUnaligned::_transformVector)

	/* --- Operators --- */
	.def(bp::self * bp::self)
	.def(bp::self *= bp::self)
	.def(bp::self * bp::other<Vector3>())
	.def("__eq__",&QuaternionVisitor::__eq__)
	.def("__ne__",&QuaternionVisitor::__ne__)
	.def("__abs__",&QuaternionUnaligned::norm)
	.def("__len__",&QuaternionVisitor::__len__).staticmethod("__len__")
	.def("__setitem__",&QuaternionVisitor::__setitem__)
	.def("__getitem__",&QuaternionVisitor::__getitem__)
	;
    }
  private:

    static QuaternionUnaligned* fromTwoVectors(const Vector3& u, const Vector3& v)
    { 
      QuaternionUnaligned* q(new QuaternionUnaligned); q->setFromTwoVectors(u,v); 
      return q; 
    }
  
    static Coefficients coeffs(const QuaternionUnaligned& self)
    {
      return self.coeffs(); 
    }

    static void setFromTwoVectors(QuaternionUnaligned& self, const Vector3& u, const Vector3& v)
    { 
      self.setFromTwoVectors(u,v);
    }

    static bool __eq__(const QuaternionUnaligned& u, const QuaternionUnaligned& v)
    {
      return u.isApprox(v,1e-9);
    }
    static bool __ne__(const QuaternionUnaligned& u, const QuaternionUnaligned& v)
    {
      return !__eq__(u,v); 
    }

    static double __getitem__(const QuaternionUnaligned & self, int idx)
    { 
      if((idx<0) || (idx>4)) throw eigenpy::ExceptionIndex(idx,0,4);
      if(idx==0) return self.x(); 
      else if(idx==1) return self.y(); 
      else if(idx==2) return self.z();
      else return self.w(); 
    }
  
    static void __setitem__(QuaternionUnaligned& self, int idx, double value)
    { 
      if((idx<0) || (idx>4)) throw eigenpy::ExceptionIndex(idx,0,4);
      if(idx==0) self.x()=value; 
      else if(idx==1) self.y()=value;
      else if(idx==2) self.z()=value;
      else self.w()=value;
    }

    static int __len__() {    return 4;  }

  public:

    static void expose()
    {
      bp::class_<QuaternionUnaligned>("Quaternion",
				      "Quaternion representing rotation.\n\n"
				      "Supported operations "
				      "('q is a Quaternion, 'v' is a Vector3): "
				      "'q*q' (rotation composition), "
				      "'q*=q', "
				      "'q*v' (rotating 'v' by 'q'), "
				      "'q==q', 'q!=q', 'q[0..3]'.",
				      bp::init<>())
	.def(QuaternionVisitor<Quaternion>())
	;
   
#ifndef EIGENPY_ALIGNED
      bp::to_python_converter< Quaternion,QuaternionVisitor<Quaternion> >();
#endif
    }

  };

} // namespace eigenpy

#endif // ifndef __eigenpy_quaternion_hpp__
