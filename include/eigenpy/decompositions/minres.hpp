/*
 * Copyright 2021 INRIA
 */

#ifndef __eigenpy_decomposition_minres_hpp__
#define __eigenpy_decomposition_minres_hpp__

#include <Eigen/Core>
#include <iostream>
#include <unsupported/Eigen/IterativeSolvers>

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {
template <typename _Solver>
struct IterativeSolverBaseVisitor
    : public boost::python::def_visitor<IterativeSolverBaseVisitor<_Solver> > {
  typedef _Solver Solver;
  typedef typename Solver::MatrixType MatrixType;
  typedef typename Solver::Preconditioner Preconditioner;
  typedef typename Solver::Scalar Scalar;
  typedef typename Solver::RealScalar RealScalar;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                        MatrixType::Options>
      MatrixXs;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def("analyzePattern",
           (Solver & (Solver::*)(const Eigen::EigenBase<MatrixType>& matrix)) &
               Solver::analyzePattern,
           bp::args("self", "A"),
           "Initializes the iterative solver for the sparsity pattern of the "
           "matrix A for further solving Ax=b problems.",
           bp::return_self<>())
        .def(
            "factorize",
            (Solver & (Solver::*)(const Eigen::EigenBase<MatrixType>& matrix)) &
                Solver::factorize,
            bp::args("self", "A"),
            "Initializes the iterative solver with the numerical values of the "
            "matrix A for further solving Ax=b problems.",
            bp::return_self<>())
        .def(
            "compute",
            (Solver & (Solver::*)(const Eigen::EigenBase<MatrixType>& matrix)) &
                Solver::compute,
            bp::args("self", "A"),
            "Initializes the iterative solver with the matrix A for further "
            "solving Ax=b problems.",
            bp::return_self<>())

        .def("rows", &Solver::rows, bp::arg("self"),
             "Returns the number of rows.")
        .def("cols", &Solver::cols, bp::arg("self"),
             "Returns the number of columns.")
        .def("tolerance", &Solver::tolerance, bp::arg("self"),
             "Returns the tolerance threshold used by the stopping criteria.")
        .def("setTolerance", &Solver::setTolerance,
             bp::args("self", "tolerance"),
             "Sets the tolerance threshold used by the stopping criteria.\n"
             "This value is used as an upper bound to the relative residual "
             "error: |Ax-b|/|b|.\n"
             "The default value is the machine precision given by "
             "NumTraits<Scalar>::epsilon().",
             bp::return_self<>())
        .def("preconditioner",
             (Preconditioner & (Solver::*)()) & Solver::preconditioner,
             bp::arg("self"),
             "Returns a read-write reference to the preconditioner for custom "
             "configuration.",
             bp::return_internal_reference<>())

        .def("maxIterations", &Solver::maxIterations, bp::arg("self"),
             "Returns the max number of iterations.\n"
             "It is either the value setted by setMaxIterations or, by "
             "default, twice the number of columns of the matrix.")
        .def("setMaxIterations", &Solver::setMaxIterations,
             bp::args("self", "max_iterations"),
             "Sets the max number of iterations.\n"
             "Default is twice the number of columns of the matrix.",
             bp::return_self<>())

        .def(
            "iterations", &Solver::iterations, bp::arg("self"),
            "Returns the number of iterations performed during the last solve.")
        .def("error", &Solver::error, bp::arg("self"),
             "Returns the tolerance error reached during the last solve.\n"
             "It is a close approximation of the true relative residual error "
             "|Ax-b|/|b|.")
        .def("info", &Solver::error, bp::arg("info"),
             "Returns Success if the iterations converged, and NoConvergence "
             "otherwise.")

        .def("solveWithGuess", &solveWithGuess<MatrixXs, MatrixXs>,
             bp::args("self", "b", "x0"),
             "Returns the solution x of A x = b using the current "
             "decomposition of A and x0 as an initial solution.")

        .def(
            "solve", &solve<MatrixXs>, bp::args("self", "b"),
            "Returns the solution x of A x = b using the current decomposition "
            "of A where b is a right hand side matrix or vector.");
  }

 private:
  template <typename MatrixOrVector1, typename MatrixOrVector2>
  static MatrixOrVector1 solveWithGuess(const Solver& self,
                                        const MatrixOrVector1& b,
                                        const MatrixOrVector2& guess) {
    return self.solveWithGuess(b, guess);
  }

  template <typename MatrixOrVector>
  static MatrixOrVector solve(const Solver& self,
                              const MatrixOrVector& mat_or_vec) {
    MatrixOrVector res = self.solve(mat_or_vec);
    return res;
  }
};

template <typename _MatrixType>
struct MINRESSolverVisitor
    : public boost::python::def_visitor<MINRESSolverVisitor<_MatrixType> > {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>
      VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                        MatrixType::Options>
      MatrixXs;
  typedef Eigen::MINRES<MatrixType> Solver;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(bp::init<>(bp::arg("self"), "Default constructor"))
        .def(bp::init<MatrixType>(
            bp::args("self", "matrix"),
            "Initialize the solver with matrix A for further Ax=b solving.\n"
            "This constructor is a shortcut for the default constructor "
            "followed by a call to compute()."))

        .def(IterativeSolverBaseVisitor<Solver>());
  }

  static void expose() {
    static const std::string classname =
        "MINRES" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string& name) {
    bp::class_<Solver, boost::noncopyable>(
        name.c_str(),
        "A minimal residual solver for sparse symmetric problems.\n"
        "This class allows to solve for A.x = b sparse linear problems using "
        "the MINRES algorithm of Paige and Saunders (1975). The sparse matrix "
        "A must be symmetric (possibly indefinite). The vectors x and b can be "
        "either dense or sparse.\n"
        "The maximal number of iterations and tolerance value can be "
        "controlled via the setMaxIterations() and setTolerance() methods. The "
        "defaults are the size of the problem for the maximal number of "
        "iterations and NumTraits<Scalar>::epsilon() for the tolerance.\n",
        bp::no_init)
        .def(MINRESSolverVisitor());
  }

 private:
  template <typename MatrixOrVector>
  static MatrixOrVector solve(const Solver& self, const MatrixOrVector& vec) {
    return self.solve(vec);
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decomposition_minres_hpp__
