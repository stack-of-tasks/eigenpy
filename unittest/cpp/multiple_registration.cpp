#include "eigenpy/registration.hpp"
#include <cstdio>

namespace bp = boost::python;

class X {
 public:
  X() {}
  void operator()() { printf("DOOT\n"); }
};

class X_wrapper : public X, bp::wrapper<X> {
 public:
  static void expose() {
    if (!eigenpy::register_symbolic_link_to_registered_type<X>()) {
      bp::class_<X>("X", bp::init<>()).def("__call__", &X::operator());
    }
  }
};

BOOST_PYTHON_MODULE(multiple_registration) {
  X_wrapper::expose();
  X_wrapper::expose();
  X_wrapper::expose();
  X_wrapper::expose();
  X_wrapper::expose();
  X_wrapper::expose();
}
