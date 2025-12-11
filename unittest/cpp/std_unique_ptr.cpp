/// @file
/// @copyright Copyright 2023 CNRS INRIA

#include <eigenpy/eigenpy.hpp>
#include <eigenpy/std-unique-ptr.hpp>

#include <memory>
#include <string>
#include <complex>

namespace bp = boost::python;

struct V1 {
  V1() = default;
  V1(double p_v) : v(p_v) {}

  double v = 100;
};

std::unique_ptr<int> make_unique_int() { return std::make_unique<int>(10); }

std::unique_ptr<V1> make_unique_v1() { return std::make_unique<V1>(10); }

std::unique_ptr<V1> make_unique_null() { return nullptr; }

std::unique_ptr<std::string> make_unique_str() {
  return std::make_unique<std::string>("str");
}

std::unique_ptr<std::complex<double>> make_unique_complex() {
  return std::make_unique<std::complex<double>>(1., 0.);
}

struct UniquePtrHolder {
  UniquePtrHolder()
      : int_ptr(std::make_unique<int>(20)),
        v1_ptr(std::make_unique<V1>(200)),
        str_ptr(std::make_unique<std::string>("str")),
        complex_ptr(std::make_unique<std::complex<double>>(1., 0.)) {}

  std::unique_ptr<int> int_ptr;
  std::unique_ptr<V1> v1_ptr;
  std::unique_ptr<V1> null_ptr;
  std::unique_ptr<std::string> str_ptr;
  std::unique_ptr<std::complex<double>> complex_ptr;
};

BOOST_PYTHON_MODULE(std_unique_ptr) {
  eigenpy::enableEigenPy();

  bp::class_<V1>("V1", bp::init<>()).def_readwrite("v", &V1::v);

  bp::def("make_unique_int", make_unique_int);
  bp::def("make_unique_v1", make_unique_v1);
  bp::def("make_unique_null", make_unique_null,
          eigenpy::StdUniquePtrCallPolicies());
  bp::def("make_unique_str", make_unique_str);
  bp::def("make_unique_complex", make_unique_complex);

  boost::python::class_<UniquePtrHolder, boost::noncopyable>("UniquePtrHolder",
                                                             bp::init<>())
      .add_property("int_ptr",
                    bp::make_getter(&UniquePtrHolder::int_ptr,
                                    eigenpy::ReturnInternalStdUniquePtr()))
      .add_property("v1_ptr",
                    bp::make_getter(&UniquePtrHolder::v1_ptr,
                                    eigenpy::ReturnInternalStdUniquePtr()))
      .add_property("null_ptr",
                    bp::make_getter(&UniquePtrHolder::null_ptr,
                                    eigenpy::ReturnInternalStdUniquePtr()))
      .add_property("str_ptr",
                    bp::make_getter(&UniquePtrHolder::str_ptr,
                                    eigenpy::ReturnInternalStdUniquePtr()))
      .add_property("complex_ptr",
                    bp::make_getter(&UniquePtrHolder::complex_ptr,
                                    eigenpy::ReturnInternalStdUniquePtr()));
}
