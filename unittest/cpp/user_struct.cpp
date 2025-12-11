#include "eigenpy/eigenpy.hpp"

struct mystruct {
  Eigen::Vector3d x_;
  Eigen::Vector4d y_;

  mystruct(const Eigen::Vector3d& x, const Eigen::Vector4d& y) : x_(x), y_(y) {}
};

BOOST_PYTHON_MODULE(user_struct) {
  using namespace Eigen;
  namespace bp = boost::python;
  eigenpy::enableEigenPy();
  bp::class_<mystruct>("MyStruct", bp::init<const Vector3d&, const Vector4d&>())
      .add_property(
          "x",
          bp::make_getter(&mystruct::x_, bp::return_internal_reference<>()),
          bp::make_setter(&mystruct::x_))
      .add_property(
          "y",
          bp::make_getter(&mystruct::y_, bp::return_internal_reference<>()),
          bp::make_setter(&mystruct::y_));
}
