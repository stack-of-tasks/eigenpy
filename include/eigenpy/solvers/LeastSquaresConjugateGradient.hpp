/*
 * Copyright 2017-2018, Justin Carpentier, LAAS-CNRS
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

#ifndef __eigenpy_least_square_conjugate_gradient_hpp__
#define __eigenpy_least_square_conjugate_gradient_hpp__

#include <Eigen/IterativeLinearSolvers>

#include "eigenpy/fwd.hpp"
#include "eigenpy/solvers/IterativeSolverBase.hpp"

namespace eigenpy {

template <typename LeastSquaresConjugateGradient>
struct LeastSquaresConjugateGradientVisitor
    : public boost::python::def_visitor<LeastSquaresConjugateGradientVisitor<
          LeastSquaresConjugateGradient> > {
  typedef Eigen::MatrixXd MatrixType;

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

  static void expose() {
    bp::class_<LeastSquaresConjugateGradient, boost::noncopyable>(
        "LeastSquaresConjugateGradient", bp::no_init)
        .def(IterativeSolverVisitor<LeastSquaresConjugateGradient>())
        .def(LeastSquaresConjugateGradientVisitor<
             LeastSquaresConjugateGradient>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_least_square_conjugate_gradient_hpp__
