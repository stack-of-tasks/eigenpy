/*
 * Copyright 2014-2023 CNRS INRIA
 */

#ifndef __eigenpy_angle_axis_hpp__
#define __eigenpy_angle_axis_hpp__

#include "eigenpy/eigenpy.hpp"

namespace eigenpy {

template <typename AngleAxis>
class AngleAxisVisitor;

template <typename Scalar>
struct call<Eigen::AngleAxis<Scalar> > {
  typedef Eigen::AngleAxis<Scalar> AngleAxis;

  static inline void expose() { AngleAxisVisitor<AngleAxis>::expose(); }

  static inline bool isApprox(
      const AngleAxis& self, const AngleAxis& other,
      const Scalar& prec = Eigen::NumTraits<Scalar>::dummy_precision()) {
    return self.isApprox(other, prec);
  }
};

template <typename AngleAxis>
class AngleAxisVisitor : public bp::def_visitor<AngleAxisVisitor<AngleAxis> > {
  typedef typename AngleAxis::Scalar Scalar;
  typedef typename AngleAxis::Vector3 Vector3;
  typedef typename AngleAxis::Matrix3 Matrix3;

  typedef typename Eigen::Quaternion<Scalar, 0> Quaternion;
  typedef Eigen::RotationBase<AngleAxis, 3> RotationBase;

  BOOST_PYTHON_FUNCTION_OVERLOADS(isApproxAngleAxis_overload,
                                  call<AngleAxis>::isApprox, 2, 3)

 public:
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(bp::init<>(bp::arg("self"), "Default constructor"))
        .def(bp::init<Scalar, Vector3>(bp::args("self", "angle", "axis"),
                                       "Initialize from angle and axis."))
        .def(bp::init<Matrix3>(bp::args("self", "R"),
                               "Initialize from a rotation matrix"))
        .def(bp::init<Quaternion>(bp::args("self", "quaternion"),
                                  "Initialize from a quaternion."))
        .def(bp::init<AngleAxis>(bp::args("self", "copy"), "Copy constructor."))

        /* --- Properties --- */
        .add_property(
            "axis",
            bp::make_function((Vector3 & (AngleAxis::*)()) & AngleAxis::axis,
                              bp::return_internal_reference<>()),
            &AngleAxisVisitor::setAxis, "The rotation axis.")
        .add_property("angle",
                      (Scalar(AngleAxis::*)() const) & AngleAxis::angle,
                      &AngleAxisVisitor::setAngle, "The rotation angle.")

        /* --- Methods --- */
        .def("inverse", &AngleAxis::inverse, bp::arg("self"),
             "Return the inverse rotation.")
        .def("fromRotationMatrix",
             &AngleAxis::template fromRotationMatrix<Matrix3>,
             (bp::arg("self"), bp::arg("rotation matrix")),
             "Sets *this from a 3x3 rotation matrix", bp::return_self<>())
        .def("toRotationMatrix", &AngleAxis::toRotationMatrix,
             //           bp::arg("self"),
             "Constructs and returns an equivalent rotation matrix.")
        .def("matrix", &AngleAxis::matrix, bp::arg("self"),
             "Returns an equivalent rotation matrix.")

        .def("isApprox", &call<AngleAxis>::isApprox,
             isApproxAngleAxis_overload(
                 bp::args("self", "other", "prec"),
                 "Returns true if *this is approximately equal to other, "
                 "within the precision determined by prec."))

        /* --- Operators --- */
        .def(bp::self * bp::other<Vector3>())
        .def(bp::self * bp::other<Quaternion>())
        .def(bp::self * bp::self)
        .def("__eq__", &AngleAxisVisitor::__eq__)
        .def("__ne__", &AngleAxisVisitor::__ne__)

        .def("__str__", &print)
        .def("__repr__", &print);
  }

 private:
  static void setAxis(AngleAxis& self, const Vector3& axis) {
    self.axis() = axis;
  }

  static void setAngle(AngleAxis& self, const Scalar& angle) {
    self.angle() = angle;
  }

  static bool __eq__(const AngleAxis& u, const AngleAxis& v) {
    return u.axis() == v.axis() && v.angle() == u.angle();
  }

  static bool __ne__(const AngleAxis& u, const AngleAxis& v) {
    return !__eq__(u, v);
  }

  static std::string print(const AngleAxis& self) {
    std::stringstream ss;
    ss << "angle: " << self.angle() << std::endl;
    ss << "axis: " << self.axis().transpose() << std::endl;

    return ss.str();
  }

 public:
  static void expose() {
    bp::class_<AngleAxis>(
        "AngleAxis", "AngleAxis representation of a rotation.\n\n", bp::no_init)
        .def(AngleAxisVisitor<AngleAxis>());

    // Cast to Eigen::RotationBase
    bp::implicitly_convertible<AngleAxis, RotationBase>();
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_angle_axis_hpp__
