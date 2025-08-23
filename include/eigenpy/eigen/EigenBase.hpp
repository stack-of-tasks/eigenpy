/*
 * Copyright 2024 INRIA
 */

#ifndef __eigenpy_eigen_eigen_base_hpp__
#define __eigenpy_eigen_eigen_base_hpp__

#include "eigenpy/eigenpy.hpp"

namespace eigenpy {

template <typename Derived>
struct EigenBaseVisitor
    : public boost::python::def_visitor<EigenBaseVisitor<Derived>> {
  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def("cols", &Derived::cols, bp::arg("self"),
           "Returns the number of columns.")
        .def("rows", &Derived::rows, bp::arg("self"),
             "Returns the number of rows.")
        .def("size", &Derived::rows, bp::arg("self"),
             "Returns the number of coefficients, which is rows()*cols().");
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_eigen_eigen_base_hpp__
