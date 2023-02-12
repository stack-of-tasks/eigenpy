//
// Copyright (c) 2016-2021 CNRS INRIA
//

#ifndef __eigenpy_utils_copyable_hpp__
#define __eigenpy_utils_copyable_hpp__

#include <boost/python.hpp>

namespace eigenpy {

///
/// \brief Add the Python method copy to allow a copy of this by calling the
/// copy constructor.
///
template <class C>
struct CopyableVisitor : public bp::def_visitor<CopyableVisitor<C> > {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def("copy", &copy, bp::arg("self"), "Returns a copy of *this.");
  }

 private:
  static C copy(const C& self) { return C(self); }
};
}  // namespace eigenpy

#endif  // ifndef __eigenpy_utils_copyable_hpp__
