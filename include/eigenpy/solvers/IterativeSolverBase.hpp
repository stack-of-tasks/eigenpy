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

#ifndef __eigenpy_iterative_solver_base_hpp__
#define __eigenpy_iterative_solver_base_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/solvers/SparseSolverBase.hpp"

namespace eigenpy {

template <typename IterativeSolver>
struct IterativeSolverVisitor : public boost::python::def_visitor<
                                    IterativeSolverVisitor<IterativeSolver> > {
  typedef typename IterativeSolver::MatrixType MatrixType;
  typedef typename IterativeSolver::Preconditioner Preconditioner;
  typedef Eigen::VectorXd VectorType;

  template <class PyClass>
  void visit(PyClass& cl) const {
    typedef IterativeSolver IS;

    cl.def(SparseSolverVisitor<IS>())
        .def("error", &IS::error,
             "Returns the tolerance error reached during the last solve.\n"
             "It is a close approximation of the true relative residual error "
             "|Ax-b|/|b|.")
        .def("info", &IS::info,
             "Returns success if the iterations converged, and NoConvergence "
             "otherwise.")
        .def(
            "iterations", &IS::iterations,
            "Returns the number of iterations performed during the last solve.")
        .def("maxIterations", &IS::maxIterations,
             "Returns the max number of iterations.\n"
             "It is either the value setted by setMaxIterations or, by "
             "default, twice the number of columns of the matrix.")
        .def("setMaxIterations", &IS::setMaxIterations,
             "Sets the max number of iterations.\n"
             "Default is twice the number of columns of the matrix.",
             bp::return_value_policy<bp::reference_existing_object>())
        .def("tolerance", &IS::tolerance,
             "Returns he tolerance threshold used by the stopping criteria.")
        .def("setTolerance", &IS::setTolerance,
             "Sets the tolerance threshold used by the stopping criteria.\n"
             "This value is used as an upper bound to the relative residual "
             "error: |Ax-b|/|b|. The default value is the machine precision.",
             bp::return_value_policy<bp::reference_existing_object>())
        .def("analyzePattern", &analyzePattern, bp::arg("A"),
             "Initializes the iterative solver for the sparsity pattern of the "
             "matrix A for further solving Ax=b problems.\n"
             "Currently, this function mostly calls analyzePattern on the "
             "preconditioner.\n"
             "In the future we might, for instance, implement column "
             "reordering for faster matrix vector products.",
             bp::return_value_policy<bp::reference_existing_object>())
        .def("factorize", &factorize, bp::arg("A"),
             "Initializes the iterative solver with the numerical values of "
             "the matrix A for further solving Ax=b problems.\n"
             "Currently, this function mostly calls factorize on the "
             "preconditioner.",
             bp::return_value_policy<bp::reference_existing_object>())
        .def("compute", &compute, bp::arg("A"),
             "Initializes the iterative solver with the numerical values of "
             "the matrix A for further solving Ax=b problems.\n"
             "Currently, this function mostly calls factorize on the "
             "preconditioner.\n"
             "In the future we might, for instance, implement column "
             "reordering for faster matrix vector products.",
             bp::return_value_policy<bp::reference_existing_object>())
        .def("solveWithGuess", &solveWithGuess, bp::args("b", "x0"),
             "Returns the solution x of Ax = b using the current decomposition "
             "of A and x0 as an initial solution.")
        .def("preconditioner",
             (Preconditioner & (IS::*)(void)) & IS::preconditioner,
             "Returns a read-write reference to the preconditioner for custom "
             "configuration.",
             bp::return_internal_reference<>());
  }

 private:
  static IterativeSolver& factorize(IterativeSolver& self,
                                    const MatrixType& m) {
    return self.factorize(m);
  }

  static IterativeSolver& compute(IterativeSolver& self, const MatrixType& m) {
    return self.compute(m);
  }

  static IterativeSolver& analyzePattern(IterativeSolver& self,
                                         const MatrixType& m) {
    return self.analyzePattern(m);
  }

  static VectorType solveWithGuess(IterativeSolver& self,
                                   const Eigen::VectorXd& b,
                                   const Eigen::VectorXd& x0) {
    return self.solveWithGuess(b, x0);
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_iterative_solver_base_hpp__
