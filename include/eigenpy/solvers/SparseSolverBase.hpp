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

#ifndef __eigenpy_sparse_solver_base_hpp__
#define __eigenpy_sparse_solver_base_hpp__

#include "eigenpy/fwd.hpp"

namespace eigenpy {

template <typename SparseSolver>
struct SparseSolverVisitor
    : public bp::def_visitor<SparseSolverVisitor<SparseSolver> > {
  typedef Eigen::VectorXd VectorType;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def("solve", &solve, bp::arg("b"),
           "Returns the solution x of Ax = b using the current decomposition "
           "of A.");
  }

 private:
  static VectorType solve(SparseSolver& self, const VectorType& b) {
    return self.solve(b);
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_sparse_solver_base_hpp__
