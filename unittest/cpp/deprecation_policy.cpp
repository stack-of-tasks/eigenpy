#include "eigenpy/eigenpy.hpp"
#include "eigenpy/deprecation-policy.hpp"

#include <iostream>

namespace bp = boost::python;
using eigenpy::DeprecationType;

void some_deprecated_function() {
  std::cout << "Calling this should produce a warning" << std::endl;
}

void some_future_deprecated_function() {
  std::cout
      << "Calling this should produce a warning about a future deprecation"
      << std::endl;
}

class X {
 public:
  void deprecated_member_function() {}
};

BOOST_PYTHON_MODULE(deprecation_policy) {
  bp::def("some_deprecated_function", some_deprecated_function,
          eigenpy::deprecated_function<DeprecationType::DEPRECATION>());
  bp::def("some_future_deprecated_function", some_future_deprecated_function,
          eigenpy::deprecated_function<DeprecationType::FUTURE>());

  bp::class_<X>("X", bp::init<>(bp::args("self")))
      .def("deprecated_member_function", &X::deprecated_member_function,
           eigenpy::deprecated_member<>());
}
