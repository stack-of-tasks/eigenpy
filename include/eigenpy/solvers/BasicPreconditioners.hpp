/*
 * Copyright 2017, Justin Carpentier, LAAS-CNRS
 *
 * This file is part of eigenpy.
 * eigenpy is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 * eigenpy is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.  You should
 * have received a copy of the GNU Lesser General Public License along
 * with eigenpy.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __eigenpy_basic_preconditioners_hpp__
#define __eigenpy_basic_preconditioners_hpp__

#include <Eigen/IterativeLinearSolvers>

#include "eigenpy/fwd.hpp"

namespace eigenpy {

template <typename Preconditioner>
struct PreconditionerBaseVisitor
    : public bp::def_visitor<PreconditionerBaseVisitor<Preconditioner> > {
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
    : PreconditionerBaseVisitor<Eigen::DiagonalPreconditioner<Scalar> > {
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
        bp::no_init);
  }
};

#if EIGEN_VERSION_AT_LEAST(3, 3, 5)
template <typename Scalar>
struct LeastSquareDiagonalPreconditionerVisitor
    : PreconditionerBaseVisitor<
          Eigen::LeastSquareDiagonalPreconditioner<Scalar> > {
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
        .def(DiagonalPreconditionerVisitor<Scalar>());
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
        .def(PreconditionerBaseVisitor<Preconditioner>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_basic_preconditioners_hpp__
