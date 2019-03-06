/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#ifndef __eigenpy_quaternion_hpp__
#define __eigenpy_quaternion_hpp__

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "eigenpy/exception.hpp"
#include "eigenpy/registration.hpp"

namespace eigenpy
{

  class ExceptionIndex : public Exception
  {
  public:
    ExceptionIndex(int index,int imin,int imax) : Exception("")
    {
      std::ostringstream oss; oss << "Index " << index << " out of range " << imin << ".."<< imax <<".";
      message = oss.str();
    }
  };

  namespace bp = boost::python;

  template<typename Quaternion>
  class QuaternionVisitor
    :  public boost::python::def_visitor< QuaternionVisitor<Quaternion> >
  {
    typedef Eigen::QuaternionBase<Quaternion> QuaternionBase;

    typedef typename QuaternionBase::Scalar Scalar;
    typedef typename Quaternion::Coefficients Coefficients;
    typedef typename QuaternionBase::Vector3 Vector3;
    typedef typename Eigen::Matrix<Scalar,4,1> Vector4;
    typedef typename QuaternionBase::Matrix3 Matrix3;

    typedef typename QuaternionBase::AngleAxisType AngleAxis;

  public:

    template<class PyClass>
    void visit(PyClass& cl) const 
    {
      cl
      .def(bp::init<>("Default constructor"))
      .def(bp::init<Matrix3>((bp::arg("matrixRotation")),"Initialize from rotation matrix."))
      .def(bp::init<AngleAxis>((bp::arg("angleaxis")),"Initialize from angle axis."))
      .def(bp::init<Quaternion>((bp::arg("clone")),"Copy constructor."))
      .def("__init__",bp::make_constructor(&QuaternionVisitor::FromTwoVectors,
                                           bp::default_call_policies(),
                                           (bp::arg("u"),bp::arg("v"))),"Initialize from two vector u,v")
      .def(bp::init<Scalar,Scalar,Scalar,Scalar>
           ((bp::arg("w"),bp::arg("x"),bp::arg("y"),bp::arg("z")),
            "Initialize from coefficients.\n\n"
            "... note:: The order of coefficients is *w*, *x*, *y*, *z*. "
            "The [] operator numbers them differently, 0...4 for *x* *y* *z* *w*!"))
      
      .add_property("x",
                    &QuaternionVisitor::getCoeff<0>,
                    &QuaternionVisitor::setCoeff<0>,"The x coefficient.")
      .add_property("y",
                    &QuaternionVisitor::getCoeff<1>,
                    &QuaternionVisitor::setCoeff<1>,"The y coefficient.")
      .add_property("z",
                    &QuaternionVisitor::getCoeff<2>,
                    &QuaternionVisitor::setCoeff<2>,"The z coefficient.")
      .add_property("w",
                    &QuaternionVisitor::getCoeff<3>,
                    &QuaternionVisitor::setCoeff<3>,"The w coefficient.")
      
//      .def("isApprox",(bool (Quaternion::*)(const Quaternion &))&Quaternion::template isApprox<Quaternion>,
//           "Returns true if *this is approximately equal to other.")
//      .def("isApprox",(bool (Quaternion::*)(const Quaternion &, const Scalar prec))&Quaternion::template isApprox<Quaternion>,
//           "Returns true if *this is approximately equal to other, within the precision determined by prec..")
      .def("isApprox",(bool (*)(const Quaternion &))&isApprox,
           "Returns true if *this is approximately equal to other.")
      .def("isApprox",(bool (*)(const Quaternion &, const Scalar prec))&isApprox,
           "Returns true if *this is approximately equal to other, within the precision determined by prec..")
      
      /* --- Methods --- */
      .def("coeffs",(const Vector4 & (Quaternion::*)()const)&Quaternion::coeffs,
           bp::return_value_policy<bp::copy_const_reference>())
      .def("matrix",&Quaternion::matrix,"Returns an equivalent rotation matrix")
      .def("toRotationMatrix ",&Quaternion::toRotationMatrix,"Returns an equivalent 3x3 rotation matrix.")
      
      .def("setFromTwoVectors",&setFromTwoVectors,((bp::arg("a"),bp::arg("b"))),"Set *this to be the quaternion which transform a into b through a rotation."
           ,bp::return_self<>())
      .def("conjugate",&Quaternion::conjugate,"Returns the conjugated quaternion. The conjugate of a quaternion represents the opposite rotation.")
      .def("inverse",&Quaternion::inverse,"Returns the quaternion describing the inverse rotation.")
      .def("setIdentity",&Quaternion::setIdentity,bp::return_self<>(),"Set *this to the idendity rotation.")
      .def("norm",&Quaternion::norm,"Returns the norm of the quaternion's coefficients.")
      .def("normalize",&Quaternion::normalize,"Normalizes the quaternion *this.")
      .def("normalized",&Quaternion::normalized,"Returns a normalized copy of *this.")
      .def("squaredNorm",&Quaternion::squaredNorm,"Returns the squared norm of the quaternion's coefficients.")
      .def("dot",&Quaternion::template dot<Quaternion>,bp::arg("other"),"Returns the dot product of *this with other"
           "Geometrically speaking, the dot product of two unit quaternions corresponds to the cosine of half the angle between the two rotations.")
      .def("_transformVector",&Quaternion::_transformVector,bp::arg("vector"),"Rotation of a vector by a quaternion.")
      .def("vec",&vec,"Returns a vector expression of the imaginary part (x,y,z).")
      .def("angularDistance",&Quaternion::template angularDistance<Quaternion>,"Returns the angle (in radian) between two rotations.")
      .def("slerp",&slerp,bp::args("t","other"),
           "Returns the spherical linear interpolation between the two quaternions *this and other at the parameter t in [0;1].")

      /* --- Operators --- */
      .def(bp::self * bp::self)
      .def(bp::self *= bp::self)
      .def(bp::self * bp::other<Vector3>())
      .def("__eq__",&QuaternionVisitor::__eq__)
      .def("__ne__",&QuaternionVisitor::__ne__)
      .def("__abs__",&Quaternion::norm)
      .def("__len__",&QuaternionVisitor::__len__).staticmethod("__len__")
      .def("__setitem__",&QuaternionVisitor::__setitem__)
      .def("__getitem__",&QuaternionVisitor::__getitem__)
      .def("assign",&assign<Quaternion>,
           bp::arg("quat"),"Set *this from an quaternion quat and returns a reference to *this.",bp::return_self<>())
      .def("assign",(Quaternion & (Quaternion::*)(const AngleAxis &))&Quaternion::operator=,
           bp::arg("aa"),"Set *this from an angle-axis aa and returns a reference to *this.",bp::return_self<>())
      .def("__str__",&print)
      .def("__repr__",&print)
      
//      .def("FromTwoVectors",&Quaternion::template FromTwoVectors<Vector3,Vector3>,
//           bp::args("a","b"),
//           "Returns the quaternion which transform a into b through a rotation.")
      .def("FromTwoVectors",&FromTwoVectors,
           bp::args("a","b"),
           "Returns the quaternion which transform a into b through a rotation.",
           bp::return_value_policy<bp::manage_new_object>())
      .staticmethod("FromTwoVectors")
      .def("Identity",&Quaternion::Identity,"Returns a quaternion representing an identity rotation.")
      .staticmethod("Identity")
      
    
      ;
    }
  private:
    
    template<int i>
    static void setCoeff(Quaternion & self, Scalar value) { self.coeffs()[i] = value; }
    
    template<int i>
    static Scalar getCoeff(Quaternion & self) { return self.coeffs()[i]; }
    
    static Quaternion & setFromTwoVectors(Quaternion & self, const Vector3 & a, const Vector3 & b)
    { return self.setFromTwoVectors(a,b); }
    
    template<typename OtherQuat>
    static Quaternion & assign(Quaternion & self, const OtherQuat & quat)
    { return self = quat; }

    static Quaternion* FromTwoVectors(const Vector3& u, const Vector3& v)
    { 
      Quaternion* q(new Quaternion); q->setFromTwoVectors(u,v);
      return q; 
    }
    
    static bool isApprox(const Quaternion & self, const Quaternion & other,
                         const Scalar prec = Eigen::NumTraits<Scalar>::dummy_precision)
    {
      return self.isApprox(other,prec);
    }
  
    static bool __eq__(const Quaternion& u, const Quaternion& v)
    {
      return u.isApprox(v,1e-9);
    }
    
    static bool __ne__(const Quaternion& u, const Quaternion& v)
    {
      return !__eq__(u,v); 
    }

    static Scalar __getitem__(const Quaternion & self, int idx)
    { 
      if((idx<0) || (idx>=4)) throw eigenpy::ExceptionIndex(idx,0,3);
      return self.coeffs()[idx];
    }
  
    static void __setitem__(Quaternion& self, int idx, const Scalar value)
    { 
      if((idx<0) || (idx>=4)) throw eigenpy::ExceptionIndex(idx,0,3);
      self.coeffs()[idx] = value;
    }

    static int __len__() {  return 4;  }
    static Vector3 vec(const Quaternion & self) { return self.vec(); }
    
    static std::string print(const Quaternion & self)
    {
      std::stringstream ss;
      ss << "(x,y,z,w) = " << self.coeffs().transpose() << std::endl;
      
      return ss.str();
    }
    
    static Quaternion slerp(const Quaternion & self, const Scalar t, const Quaternion & other)
    { return self.slerp(t,other); }

  public:

    static void expose()
    {
      if(check_registration<Quaternion>()) return;
      
      bp::class_<Quaternion>("Quaternion",
                             "Quaternion representing rotation.\n\n"
                             "Supported operations "
                             "('q is a Quaternion, 'v' is a Vector3): "
                             "'q*q' (rotation composition), "
                             "'q*=q', "
                             "'q*v' (rotating 'v' by 'q'), "
                             "'q==q', 'q!=q', 'q[0..3]'.",
                             bp::no_init)
      .def(QuaternionVisitor<Quaternion>())
      ;
   
    }

  };

} // namespace eigenpy

#endif // ifndef __eigenpy_quaternion_hpp__
