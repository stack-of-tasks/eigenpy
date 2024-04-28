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

BOOST_PYTHON_MODULE(deprecation_policy) {
  bp::def("some_deprecated_function", some_deprecated_function,
          eigenpy::deprecation_warning_policy<DeprecationType::DEPRECATION>());
  bp::def("some_future_deprecated_function", some_future_deprecated_function,
          eigenpy::deprecation_warning_policy<DeprecationType::FUTURE>());
}
