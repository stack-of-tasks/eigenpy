/*
 * Copyright 2017 CNRS
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_solvers_conjugate_gradient_hpp__
#define __eigenpy_solvers_conjugate_gradient_hpp__

#include <Eigen/IterativeLinearSolvers>

#include "eigenpy/fwd.hpp"
#include "eigenpy/solvers/IterativeSolverBase.hpp"

namespace eigenpy {

template <typename ConjugateGradient>
struct ConjugateGradientVisitor
    : public boost::python::def_visitor<
          ConjugateGradientVisitor<ConjugateGradient>> {
  typedef typename ConjugateGradient::MatrixType MatrixType;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(IterativeSolverVisitor<ConjugateGradient>())
        .def(bp::init<>("Default constructor"))
        .def(bp::init<MatrixType>(
            bp::arg("A"),
            "Initialize the solver with matrix A for further Ax=b solving.\n"
            "This constructor is a shortcut for the default constructor "
            "followed by a call to compute()."));
  }

  static void expose(const std::string& name = "ConjugateGradient") {
    bp::class_<ConjugateGradient, boost::noncopyable>(name.c_str(), bp::no_init)
        .def(ConjugateGradientVisitor<ConjugateGradient>())
        .def(IdVisitor<ConjugateGradient>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_solvers_conjugate_gradient_hpp__
