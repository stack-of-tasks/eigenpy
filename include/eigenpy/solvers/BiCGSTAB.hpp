/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_solvers_bicgstab_hpp__
#define __eigenpy_solvers_bicgstab_hpp__

#include <Eigen/IterativeLinearSolvers>

#include "eigenpy/fwd.hpp"
#include "eigenpy/solvers/IterativeSolverBase.hpp"

namespace eigenpy {

template <typename BiCGSTAB>
struct BiCGSTABVisitor
    : public boost::python::def_visitor<BiCGSTABVisitor<BiCGSTAB>> {
  typedef typename BiCGSTAB::MatrixType MatrixType;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(bp::init<>("Default constructor"))
        .def(bp::init<MatrixType>(
            bp::arg("A"),
            "Initialize the solver with matrix A for further || Ax - b || "
            "solving.\n"
            "This constructor is a shortcut for the default constructor "
            "followed by a call to compute()."));
  }

  static void expose(const std::string& name = "BiCGSTAB") {
    bp::class_<BiCGSTAB, boost::noncopyable>(name.c_str(), bp::no_init)
        .def(IterativeSolverVisitor<BiCGSTAB>())
        .def(BiCGSTABVisitor<BiCGSTAB>())
        .def(IdVisitor<BiCGSTAB>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_solvers_bicgstab_hpp__
