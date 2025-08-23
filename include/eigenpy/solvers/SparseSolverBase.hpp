/*
 * Copyright 2017 CNRS
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_solvers_sparse_solver_base_hpp__
#define __eigenpy_solvers_sparse_solver_base_hpp__

#include "eigenpy/fwd.hpp"

namespace eigenpy {

template <typename SparseSolver>
struct SparseSolverVisitor
    : public bp::def_visitor<SparseSolverVisitor<SparseSolver>> {
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

#endif  // ifndef __eigenpy_solvers_sparse_solver_base_hpp__
