/*
 * Copyright 2017-2018 CNRS
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_solvers_least_square_conjugate_gradient_hpp__
#define __eigenpy_solvers_least_square_conjugate_gradient_hpp__

#include <Eigen/IterativeLinearSolvers>

#include "eigenpy/fwd.hpp"
#include "eigenpy/solvers/IterativeSolverBase.hpp"

namespace eigenpy {

template <typename LeastSquaresConjugateGradient>
struct LeastSquaresConjugateGradientVisitor
    : public boost::python::def_visitor<
          LeastSquaresConjugateGradientVisitor<LeastSquaresConjugateGradient>> {
  typedef typename LeastSquaresConjugateGradient::MatrixType MatrixType;

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

  static void expose(
      const std::string& name = "LeastSquaresConjugateGradient") {
    bp::class_<LeastSquaresConjugateGradient, boost::noncopyable>(name.c_str(),
                                                                  bp::no_init)
        .def(IterativeSolverVisitor<LeastSquaresConjugateGradient>())
        .def(LeastSquaresConjugateGradientVisitor<
             LeastSquaresConjugateGradient>())
        .def(IdVisitor<LeastSquaresConjugateGradient>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_solvers_least_square_conjugate_gradient_hpp__
