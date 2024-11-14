///
/// Copyright 2024 INRIA
///

#include <typeinfo>
#include <typeindex>

#include <boost/python.hpp>
#include <boost/type_index.hpp>

#include "eigenpy/registration.hpp"

namespace bp = boost::python;

namespace eigenpy {

void exposeStdTypeIndex() {
  typedef std::type_index Self;
  if (register_symbolic_link_to_registered_type<Self>()) return;

  bp::class_<Self>(
      "std_type_index",
      "The class type_index holds implementation-specific information about a "
      "type, including the name of the type and means to compare two types for "
      "equality or collating order.",
      bp::no_init)
      .def(bp::self == bp::self)
      .def(bp::self >= bp::self)
      .def(bp::self > bp::self)
      .def(bp::self < bp::self)
      .def(bp::self <= bp::self)
      .def("hash_code", &Self::hash_code, bp::arg("self"),
           "Returns an unspecified value (here denoted by hash code) such that "
           "for all std::type_info objects referring to the same type, their "
           "hash code is the same.")
      .def("name", &Self::name, bp::arg("self"),
           "Returns an implementation defined null-terminated character string "
           "containing the name of the type. No guarantees are given; in "
           "particular, the returned string can be identical for several types "
           "and change between invocations of the same program.")
      .def(
          "pretty_name",
          +[](const Self &value) -> std::string {
            return boost::core::demangle(value.name());
          },
          bp::arg("self"), "Human readible name.");
}

void exposeBoostTypeIndex() {
  typedef boost::typeindex::type_index Self;
  if (register_symbolic_link_to_registered_type<Self>()) return;

  bp::class_<Self>(
      "boost_type_index",
      "The class type_index holds implementation-specific information about a "
      "type, including the name of the type and means to compare two types for "
      "equality or collating order.",
      bp::no_init)
      .def(bp::self == bp::self)
      .def(bp::self >= bp::self)
      .def(bp::self > bp::self)
      .def(bp::self < bp::self)
      .def(bp::self <= bp::self)
      .def("hash_code", &Self::hash_code, bp::arg("self"),
           "Returns an unspecified value (here denoted by hash code) such that "
           "for all std::type_info objects referring to the same type, their "
           "hash code is the same.")
      .def("name", &Self::name, bp::arg("self"),
           "Returns an implementation defined null-terminated character string "
           "containing the name of the type. No guarantees are given; in "
           "particular, the returned string can be identical for several types "
           "and change between invocations of the same program.")
      .def("pretty_name", &Self::pretty_name, bp::arg("self"),
           "Human readible name.");
}

void exposeTypeInfo() {
  exposeStdTypeIndex();
  exposeBoostTypeIndex();
}
}  // namespace eigenpy
