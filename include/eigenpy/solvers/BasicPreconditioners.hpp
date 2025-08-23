/*
 * Copyright 2017 CNRS
 * Copyright 2024-2025 INRIA
 */

#ifndef __eigenpy_solvers_basic_preconditioners_hpp__
#define __eigenpy_solvers_basic_preconditioners_hpp__

#include <Eigen/IterativeLinearSolvers>

#include "eigenpy/fwd.hpp"

namespace eigenpy {

template <typename Preconditioner>
struct PreconditionerBaseVisitor
    : public bp::def_visitor<PreconditionerBaseVisitor<Preconditioner>> {
  typedef Eigen::MatrixXd MatrixType;
  typedef Eigen::VectorXd VectorType;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(bp::init<>("Default constructor"))
        .def(bp::init<MatrixType>(bp::args("self", "A"),
                                  "Initialize the preconditioner with matrix A "
                                  "for further Az=b solving."))
#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
        .def("info", &Preconditioner::info,
             "Returns success if the Preconditioner has been well initialized.")
#endif
        .def("solve", &solve, bp::arg("b"),
             "Returns the solution A * z = b where the preconditioner is an "
             "estimate of A^-1.")

        .def("compute", &Preconditioner::template compute<MatrixType>,
             bp::arg("mat"),
             "Initialize the preconditioner from the matrix value.",
             bp::return_value_policy<bp::reference_existing_object>())
        .def("factorize", &Preconditioner::template factorize<MatrixType>,
             bp::arg("mat"),
             "Initialize the preconditioner from the matrix value, i.e "
             "factorize the mat given as input to approximate its inverse.",
             bp::return_value_policy<bp::reference_existing_object>());
  }

 private:
  static VectorType solve(Preconditioner& self, const VectorType& b) {
    return self.solve(b);
  }
};

template <typename Scalar>
struct DiagonalPreconditionerVisitor
    : PreconditionerBaseVisitor<Eigen::DiagonalPreconditioner<Scalar>> {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
  typedef Eigen::DiagonalPreconditioner<Scalar> Preconditioner;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(PreconditionerBaseVisitor<Preconditioner>())
        .def("rows", &Preconditioner::rows,
             "Returns the number of rows in the preconditioner.")
        .def("cols", &Preconditioner::cols,
             "Returns the number of cols in the preconditioner.");
  }

  static void expose() {
    bp::class_<Preconditioner>(
        "DiagonalPreconditioner",
        "A preconditioner based on the digonal entrie.\n"
        "This class allows to approximately solve for A.x = b problems "
        "assuming A is a diagonal matrix.",
        bp::no_init)
        .def(IdVisitor<Preconditioner>());
  }
};

#if EIGEN_VERSION_AT_LEAST(3, 3, 5)
template <typename Scalar>
struct LeastSquareDiagonalPreconditionerVisitor
    : PreconditionerBaseVisitor<
          Eigen::LeastSquareDiagonalPreconditioner<Scalar>> {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
  typedef Eigen::LeastSquareDiagonalPreconditioner<Scalar> Preconditioner;

  template <class PyClass>
  void visit(PyClass&) const {}

  static void expose() {
    bp::class_<Preconditioner>(
        "LeastSquareDiagonalPreconditioner",
        "Jacobi preconditioner for LeastSquaresConjugateGradient.\n"
        "his class allows to approximately solve for A' A x  = A' b problems "
        "assuming A' A is a diagonal matrix.",
        bp::no_init)
        .def(DiagonalPreconditionerVisitor<Scalar>())
        .def(IdVisitor<Preconditioner>());
  }
};
#endif

struct IdentityPreconditionerVisitor
    : PreconditionerBaseVisitor<Eigen::IdentityPreconditioner> {
  typedef Eigen::IdentityPreconditioner Preconditioner;

  template <class PyClass>
  void visit(PyClass&) const {}

  static void expose() {
    bp::class_<Preconditioner>("IdentityPreconditioner", bp::no_init)
        .def(PreconditionerBaseVisitor<Preconditioner>())
        .def(IdVisitor<Preconditioner>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_solvers_basic_preconditioners_hpp__
