//
// Copyright (c) 2024 INRIA
//

#ifndef __eigenpy_id_hpp__
#define __eigenpy_id_hpp__

#include <boost/python.hpp>
#include <boost/cstdint.hpp>

namespace eigenpy {

///
/// \brief Add the Python method id to retrieving a unique id for a given object
/// exposed with Boost.Python
///
template <class C>
struct IdVisitor : public bp::def_visitor<IdVisitor<C>> {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def("id", &id, bp::arg("self"),
           "Returns the unique identity of an object.\n"
           "For object held in C++, it corresponds to its memory address.");
  }

 private:
  static boost::int64_t id(const C& self) {
    return boost::int64_t(reinterpret_cast<const void*>(&self));
  }
};
}  // namespace eigenpy

#endif  // ifndef __eigenpy_id_hpp__
