/// @file
/// @copyright Copyright 2023 CNRS INRIA

#include <iostream>
#include <eigenpy/eigenpy.hpp>
#include <eigenpy/std_unique_ptr.hpp>
#include <memory>

namespace bp = boost::python;

struct V1 {
  V1() = default;
  V1(double p_v) : v(p_v) {}

  double v = 100;
};

std::unique_ptr<int> make_unique_int() { return std::make_unique<int>(10); }

std::unique_ptr<V1> make_unique_v1() { return std::make_unique<V1>(10); }

std::unique_ptr<V1> make_unique_null() { return nullptr; }

struct UniquePtrHolder {
  std::unique_ptr<int> int_ptr;
  std::unique_ptr<V1> v1_ptr;
  std::unique_ptr<V1> null_ptr;
};

BOOST_PYTHON_MODULE(std_unique_ptr) {
  eigenpy::enableEigenPy();

  bp::class_<V1>("V1", bp::init<>()).def_readwrite("v", &V1::v);

  bp::def("make_unique_int", make_unique_int,
          eigenpy::StdUniquePtrCallPolicies());
  bp::def("make_unique_v1", make_unique_v1,
          eigenpy::StdUniquePtrCallPolicies());
  bp::def("make_unique_null", make_unique_null,
          eigenpy::StdUniquePtrCallPolicies());
  // TODO allow access with a CallPolicie like return_internal_reference
  // boost::python::class_<UniquePtrHolder>("UniquePtrHolder", bp::init<>())
  //     .add_property("int_ptr", bp::make_getter(&UniquePtrHolder::int_ptr),
  //                   bp::make_setter(&UniquePtrHolder::int_ptr));
}
