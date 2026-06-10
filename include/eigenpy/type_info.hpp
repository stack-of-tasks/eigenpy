///
/// Copyright (c) 2024 INRIA
///

#ifndef __eigenpy_type_info_hpp__
#define __eigenpy_type_info_hpp__

#include "eigenpy/fwd.hpp"

#include <boost/type_index.hpp>
#include <typeinfo>
#include <typeindex>

namespace eigenpy {

template <typename T>
boost::typeindex::type_index type_info(const T& value) {
  return boost::typeindex::type_id_runtime(value);
}

template <typename T>
void expose_boost_type_info() {
  boost::python::def(
      "type_info",
      +[](const T& value) -> boost::typeindex::type_index {
        return boost::typeindex::type_id_runtime(value);
      },
      bp::arg("value"),
      "Returns information of the type of value as a "
      "boost::typeindex::type_index (can work without RTTI).");
  boost::python::def(
      "boost_type_info",
      +[](const T& value) -> boost::typeindex::type_index {
        return boost::typeindex::type_id_runtime(value);
      },
      bp::arg("value"),
      "Returns information of the type of value as a "
      "boost::typeindex::type_index (can work without RTTI).");
}

template <typename T>
void expose_std_type_info() {
  boost::python::def(
      "std_type_info",
      +[](const T& value) -> std::type_index { return typeid(value); },
      bp::arg("value"),
      "Returns information of the type of value as a std::type_index.");
}

///
/// \brief Add the Python method type_info to query information of a type.
///
template <class C>
struct TypeInfoVisitor : public bp::def_visitor<TypeInfoVisitor<C>> {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def("type_info", &boost_type_info, bp::arg("self"),
           "Queries information of the type of *this as a "
           "boost::typeindex::type_index (can work without RTTI).");
    cl.def("boost_type_info", &boost_type_info, bp::arg("self"),
           "Queries information of the type of *this as a "
           "boost::typeindex::type_index (can work without RTTI).");
    cl.def("std_type_info", &std_type_info, bp::arg("self"),
           "Queries information of the type of *this as a std::type_index.");
  }

 private:
  static boost::typeindex::type_index boost_type_info(const C& self) {
    return boost::typeindex::type_id_runtime(self);
  }

  static std::type_index std_type_info(const C& self) { return typeid(self); }
};

}  // namespace eigenpy

#endif  // __eigenpy_type_info_hpp__
