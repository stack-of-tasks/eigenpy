/*
 * Copyright 2014-2023 CNRS INRIA
 */

#ifndef __eigenpy_quaternion_hpp__
#define __eigenpy_quaternion_hpp__

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/exception.hpp"
#include "eigenpy/eigen-from-python.hpp"

namespace boost {
namespace python {
namespace converter {

/// \brief Template specialization of rvalue_from_python_data
template <typename Quaternion>
struct rvalue_from_python_data<Eigen::QuaternionBase<Quaternion> const&>
    : ::eigenpy::rvalue_from_python_data<Quaternion const&> {
  EIGENPY_RVALUE_FROM_PYTHON_DATA_INIT(Quaternion const&)
};

template <class Quaternion>
struct implicit<Quaternion, Eigen::QuaternionBase<Quaternion> > {
  typedef Quaternion Source;
  typedef Eigen::QuaternionBase<Quaternion> Target;

  static void* convertible(PyObject* obj) {
    // Find a converter which can produce a Source instance from
    // obj. The user has told us that Source can be converted to
    // Target, and instantiating construct() below, ensures that
    // at compile-time.
    return implicit_rvalue_convertible_from_python(
               obj, registered<Source>::converters)
               ? obj
               : 0;
  }

  static void construct(PyObject* obj, rvalue_from_python_stage1_data* data) {
    void* storage = ((rvalue_from_python_storage<Target>*)data)->storage.bytes;

    arg_from_python<Source> get_source(obj);
    bool convertible = get_source.convertible();
    BOOST_VERIFY(convertible);

    new (storage) Source(get_source());

    // record successful construction
    data->convertible = storage;
  }
};

}  // namespace converter
}  // namespace python
}  // namespace boost

namespace eigenpy {

class ExceptionIndex : public Exception {
 public:
  ExceptionIndex(int index, int imin, int imax) : Exception("") {
    std::ostringstream oss;
    oss << "Index " << index << " out of range " << imin << ".." << imax << ".";
    message = oss.str();
  }
};

template <typename QuaternionDerived>
class QuaternionVisitor;

template <typename Scalar, int Options>
struct call<Eigen::Quaternion<Scalar, Options> > {
  typedef Eigen::Quaternion<Scalar, Options> Quaternion;
  static inline void expose() { QuaternionVisitor<Quaternion>::expose(); }

  static inline bool isApprox(
      const Quaternion& self, const Quaternion& other,
      const Scalar& prec = Eigen::NumTraits<Scalar>::dummy_precision()) {
    return self.isApprox(other, prec);
  }
};

template <typename Quaternion>
class QuaternionVisitor
    : public bp::def_visitor<QuaternionVisitor<Quaternion> > {
  typedef Eigen::QuaternionBase<Quaternion> QuaternionBase;

  typedef typename QuaternionBase::Scalar Scalar;
  typedef typename Quaternion::Coefficients Coefficients;
  typedef typename QuaternionBase::Vector3 Vector3;
  typedef Coefficients Vector4;
  typedef typename QuaternionBase::Matrix3 Matrix3;

  typedef typename QuaternionBase::AngleAxisType AngleAxis;

  BOOST_PYTHON_FUNCTION_OVERLOADS(isApproxQuaternion_overload,
                                  call<Quaternion>::isApprox, 2, 3)

 public:
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def("__init__",
           bp::make_constructor(&QuaternionVisitor::FromRotationMatrix,
                                bp::default_call_policies(), (bp::arg("R"))),
           "Initialize from rotation matrix.\n"
           "\tR : a rotation matrix 3x3.")
        .def("__init__",
             bp::make_constructor(&QuaternionVisitor::FromAngleAxis,
                                  bp::default_call_policies(), (bp::arg("aa"))),
             "Initialize from an angle axis.\n"
             "\taa: angle axis object.")
        .def("__init__",
             bp::make_constructor(&QuaternionVisitor::FromOtherQuaternion,
                                  bp::default_call_policies(),
                                  (bp::arg("quat"))),
             "Copy constructor.\n"
             "\tquat: a quaternion.")
        .def("__init__",
             bp::make_constructor(&QuaternionVisitor::FromTwoVectors,
                                  bp::default_call_policies(),
                                  (bp::arg("u"), bp::arg("v"))),
             "Initialize from two vectors u and v")
        .def("__init__",
             bp::make_constructor(&QuaternionVisitor::FromOneVector,
                                  bp::default_call_policies(),
                                  (bp::arg("vec4"))),
             "Initialize from a vector 4D.\n"
             "\tvec4 : a 4D vector representing quaternion coefficients in the "
             "order xyzw.")
        .def("__init__",
             bp::make_constructor(&QuaternionVisitor::DefaultConstructor),
             "Default constructor")
        .def("__init__",
             bp::make_constructor(
                 &QuaternionVisitor::FromCoefficients,
                 bp::default_call_policies(),
                 (bp::arg("w"), bp::arg("x"), bp::arg("y"), bp::arg("z"))),
             "Initialize from coefficients.\n\n"
             "... note:: The order of coefficients is *w*, *x*, *y*, *z*. "
             "The [] operator numbers them differently, 0...4 for *x* *y* *z* "
             "*w*!")

        .add_property("x", &QuaternionVisitor::getCoeff<0>,
                      &QuaternionVisitor::setCoeff<0>, "The x coefficient.")
        .add_property("y", &QuaternionVisitor::getCoeff<1>,
                      &QuaternionVisitor::setCoeff<1>, "The y coefficient.")
        .add_property("z", &QuaternionVisitor::getCoeff<2>,
                      &QuaternionVisitor::setCoeff<2>, "The z coefficient.")
        .add_property("w", &QuaternionVisitor::getCoeff<3>,
                      &QuaternionVisitor::setCoeff<3>, "The w coefficient.")

        .def("isApprox", &call<Quaternion>::isApprox,
             isApproxQuaternion_overload(
                 bp::args("self", "other", "prec"),
                 "Returns true if *this is approximately equal to other, "
                 "within the precision determined by prec."))

        /* --- Methods --- */
        .def("coeffs",
             (const Vector4& (Quaternion::*)() const) & Quaternion::coeffs,
             bp::arg("self"), "Returns a vector of the coefficients (x,y,z,w)",
             bp::return_internal_reference<>())
        .def("matrix", &Quaternion::matrix, bp::arg("self"),
             "Returns an equivalent 3x3 rotation matrix. Similar to "
             "toRotationMatrix.")
        .def("toRotationMatrix", &Quaternion::toRotationMatrix,
             //           bp::arg("self"), // Bug in Boost.Python
             "Returns an equivalent rotation matrix.")

        .def("setFromTwoVectors", &setFromTwoVectors,
             ((bp::arg("self"), bp::arg("a"), bp::arg("b"))),
             "Set *this to be the quaternion which transforms a into b through "
             "a rotation.",
             bp::return_self<>())
        .def("conjugate", &Quaternion::conjugate, bp::arg("self"),
             "Returns the conjugated quaternion.\n"
             "The conjugate of a quaternion represents the opposite rotation.")
        .def("inverse", &Quaternion::inverse, bp::arg("self"),
             "Returns the quaternion describing the inverse rotation.")
        .def("setIdentity", &Quaternion::setIdentity, bp::arg("self"),
             "Set *this to the identity rotation.", bp::return_self<>())
        .def("norm", &Quaternion::norm, bp::arg("self"),
             "Returns the norm of the quaternion's coefficients.")
        .def("normalize", &Quaternion::normalize, bp::arg("self"),
             "Normalizes the quaternion *this.", bp::return_self<>())
        .def("normalized", &normalized, bp::arg("self"),
             "Returns a normalized copy of *this.",
             bp::return_value_policy<bp::manage_new_object>())
        .def("squaredNorm", &Quaternion::squaredNorm, bp::arg("self"),
             "Returns the squared norm of the quaternion's coefficients.")
        .def("dot", &Quaternion::template dot<Quaternion>,
             (bp::arg("self"), bp::arg("other")),
             "Returns the dot product of *this with an other Quaternion.\n"
             "Geometrically speaking, the dot product of two unit quaternions "
             "corresponds to the cosine of half the angle between the two "
             "rotations.")
        .def("_transformVector", &Quaternion::_transformVector,
             (bp::arg("self"), bp::arg("vector")),
             "Rotation of a vector by a quaternion.")
        .def("vec", &vec, bp::arg("self"),
             "Returns a vector expression of the imaginary part (x,y,z).")
        .def("angularDistance",
             //           (bp::arg("self"),bp::arg("other")), // Bug in
             //           Boost.Python
             &Quaternion::template angularDistance<Quaternion>,
             "Returns the angle (in radian) between two rotations.")
        .def("slerp", &slerp, bp::args("self", "t", "other"),
             "Returns the spherical linear interpolation between the two "
             "quaternions *this and other at the parameter t in [0;1].")

        /* --- Operators --- */
        .def(bp::self * bp::self)
        .def(bp::self *= bp::self)
        .def(bp::self * bp::other<Vector3>())
        .def("__eq__", &QuaternionVisitor::__eq__)
        .def("__ne__", &QuaternionVisitor::__ne__)
        .def("__abs__", &Quaternion::norm)
        .def("__len__", &QuaternionVisitor::__len__)
        .def("__setitem__", &QuaternionVisitor::__setitem__)
        .def("__getitem__", &QuaternionVisitor::__getitem__)
        .def("assign", &assign<Quaternion>, bp::args("self", "quat"),
             "Set *this from an quaternion quat and returns a reference to "
             "*this.",
             bp::return_self<>())
        .def(
            "assign",
            (Quaternion & (Quaternion::*)(const AngleAxis&)) &
                Quaternion::operator=,
            bp::args("self", "aa"),
            "Set *this from an angle-axis aa and returns a reference to *this.",
            bp::return_self<>())
        .def("__str__", &print)
        .def("__repr__", &print)

        //      .def("FromTwoVectors",&Quaternion::template
        //      FromTwoVectors<Vector3,Vector3>,
        //           bp::args("a","b"),
        //           "Returns the quaternion which transform a into b through a
        //           rotation.")
        .def("FromTwoVectors", &FromTwoVectors, bp::args("a", "b"),
             "Returns the quaternion which transforms a into b through a "
             "rotation.",
             bp::return_value_policy<bp::manage_new_object>())
        .staticmethod("FromTwoVectors")
        .def("Identity", &Identity,
             "Returns a quaternion representing an identity rotation.",
             bp::return_value_policy<bp::manage_new_object>())
        .staticmethod("Identity");
  }

 private:
  static Quaternion* normalized(const Quaternion& self) {
    return new Quaternion(self.normalized());
  }

  template <int i>
  static void setCoeff(Quaternion& self, Scalar value) {
    self.coeffs()[i] = value;
  }

  template <int i>
  static Scalar getCoeff(Quaternion& self) {
    return self.coeffs()[i];
  }

  static Quaternion& setFromTwoVectors(Quaternion& self, const Vector3& a,
                                       const Vector3& b) {
    return self.setFromTwoVectors(a, b);
  }

  template <typename OtherQuat>
  static Quaternion& assign(Quaternion& self, const OtherQuat& quat) {
    return self = quat;
  }

  static Quaternion* Identity() {
    Quaternion* q(new Quaternion);
    q->setIdentity();
    return q;
  }

  static Quaternion* FromCoefficients(Scalar w, Scalar x, Scalar y, Scalar z) {
    Quaternion* q(new Quaternion(w, x, y, z));
    return q;
  }

  static Quaternion* FromAngleAxis(const AngleAxis& aa) {
    Quaternion* q(new Quaternion(aa));
    return q;
  }

  static Quaternion* FromTwoVectors(const Eigen::Ref<const Vector3> u,
                                    const Eigen::Ref<const Vector3> v) {
    Quaternion* q(new Quaternion);
    q->setFromTwoVectors(u, v);
    return q;
  }

  static Quaternion* FromOtherQuaternion(const Quaternion& other) {
    Quaternion* q(new Quaternion(other));
    return q;
  }

  static Quaternion* DefaultConstructor() { return new Quaternion; }

  static Quaternion* FromOneVector(const Eigen::Ref<const Vector4> v) {
    Quaternion* q(new Quaternion(v[3], v[0], v[1], v[2]));
    return q;
  }

  static Quaternion* FromRotationMatrix(const Eigen::Ref<const Matrix3> R) {
    Quaternion* q(new Quaternion(R));
    return q;
  }

  static bool __eq__(const Quaternion& u, const Quaternion& v) {
    return u.coeffs() == v.coeffs();
  }

  static bool __ne__(const Quaternion& u, const Quaternion& v) {
    return !__eq__(u, v);
  }

  static Scalar __getitem__(const Quaternion& self, int idx) {
    if ((idx < 0) || (idx >= 4)) throw eigenpy::ExceptionIndex(idx, 0, 3);
    return self.coeffs()[idx];
  }

  static void __setitem__(Quaternion& self, int idx, const Scalar value) {
    if ((idx < 0) || (idx >= 4)) throw eigenpy::ExceptionIndex(idx, 0, 3);
    self.coeffs()[idx] = value;
  }

  static int __len__() { return 4; }
  static Vector3 vec(const Quaternion& self) { return self.vec(); }

  static std::string print(const Quaternion& self) {
    std::stringstream ss;
    ss << "(x,y,z,w) = " << self.coeffs().transpose() << std::endl;

    return ss.str();
  }

  static Quaternion slerp(const Quaternion& self, const Scalar t,
                          const Quaternion& other) {
    return self.slerp(t, other);
  }

 public:
  static void expose() {
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 6
    typedef EIGENPY_SHARED_PTR_HOLDER_TYPE(Quaternion) HolderType;
#else
    typedef ::boost::python::detail::not_specified HolderType;
#endif

    bp::class_<Quaternion, HolderType>(
        "Quaternion",
        "Quaternion representing rotation.\n\n"
        "Supported operations "
        "('q is a Quaternion, 'v' is a Vector3): "
        "'q*q' (rotation composition), "
        "'q*=q', "
        "'q*v' (rotating 'v' by 'q'), "
        "'q==q', 'q!=q', 'q[0..3]'.",
        bp::no_init)
        .def(QuaternionVisitor<Quaternion>());

    // Cast to Eigen::QuaternionBase and vice-versa
    bp::implicitly_convertible<Quaternion, QuaternionBase>();
    //      bp::implicitly_convertible<QuaternionBase,Quaternion >();
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_quaternion_hpp__
