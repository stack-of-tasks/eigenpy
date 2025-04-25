/*
 * Copyright 2024 INRIA
 */

#ifndef __eigenpy_decompositions_sparse_sparse_solver_base_hpp__
#define __eigenpy_decompositions_sparse_sparse_solver_base_hpp__

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/eigen/EigenBase.hpp"

#include <Eigen/SparseCholesky>

namespace eigenpy {

template <typename SimplicialDerived>
struct SparseSolverBaseVisitor
    : public boost::python::def_visitor<
          SparseSolverBaseVisitor<SimplicialDerived>> {
  typedef SimplicialDerived Solver;

  typedef typename SimplicialDerived::MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>
      DenseVectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                        MatrixType::Options>
      DenseMatrixXs;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def("solve", &solve<DenseVectorXs>, bp::args("self", "b"),
           "Returns the solution x of A x = b using the current "
           "decomposition of A.")
        .def("solve", &solve<DenseMatrixXs>, bp::args("self", "B"),
             "Returns the solution X of A X = B using the current "
             "decomposition of A where B is a right hand side matrix.")

        .def("solve", &solve<MatrixType>, bp::args("self", "B"),
             "Returns the solution X of A X = B using the current "
             "decomposition of A where B is a right hand side matrix.");
  }

 private:
  template <typename MatrixOrVector>
  static MatrixOrVector solve(const Solver &self, const MatrixOrVector &vec) {
    return self.solve(vec);
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_sparse_sparse_solver_base_hpp__
