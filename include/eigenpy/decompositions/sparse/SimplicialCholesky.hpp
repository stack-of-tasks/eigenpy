/*
 * Copyright 2024 INRIA
 */

#ifndef __eigenpy_decomposition_sparse_simplicial_cholesky_hpp__
#define __eigenpy_decomposition_sparse_simplicial_cholesky_hpp__

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/eigen/EigenBase.hpp"

#include <Eigen/SparseCholesky>

namespace eigenpy {

template <typename SimplicialDerived>
struct SimplicialCholeskyVisitor
    : public boost::python::def_visitor<
          SimplicialCholeskyVisitor<SimplicialDerived> > {
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
    cl.def("analyzePattern", &Solver::analyzePattern,
           bp::args("self", "matrix"),
           "Performs a symbolic decomposition on the sparcity of matrix.\n"
           "This function is particularly useful when solving for several "
           "problems having the same structure.")

        .def(EigenBaseVisitor<Solver>())

        .def("matrixL", &matrixL, bp::arg("self"),
             "Returns the lower triangular matrix L.")
        .def("matrixU", &matrixU, bp::arg("self"),
             "Returns the upper triangular matrix U.")

        .def("compute",
             (Solver & (Solver::*)(const MatrixType &matrix)) & Solver::compute,
             bp::args("self", "matrix"),
             "Computes the sparse Cholesky decomposition of a given matrix.",
             bp::return_self<>())

        .def("determinant", &Solver::determinant, bp::arg("self"),
             "Returns the determinant of the underlying matrix from the "
             "current factorization.")

        .def("factorize", &Solver::factorize, bp::args("self", "matrix"),
             "Performs a numeric decomposition of a given matrix.\n"
             "The given matrix must has the same sparcity than the matrix on "
             "which the symbolic decomposition has been performed.\n"
             "See also analyzePattern().")

        .def("info", &Solver::info, bp::arg("self"),
             "NumericalIssue if the input contains INF or NaN values or "
             "overflow occured. Returns Success otherwise.")

        .def("setShift", &Solver::setShift,
             (bp::args("self", "offset"), bp::arg("scale") = RealScalar(1)),
             "Sets the shift parameters that will be used to adjust the "
             "diagonal coefficients during the numerical factorization.\n"
             "During the numerical factorization, the diagonal coefficients "
             "are transformed by the following linear model: d_ii = offset + "
             "scale * d_ii.\n"
             "The default is the identity transformation with offset=0, and "
             "scale=1.",
             bp::return_self<>())

        .def("solve", &solve<DenseVectorXs>, bp::args("self", "b"),
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
  static MatrixType matrixL(const Solver &self) { return self.matrixL(); }
  static MatrixType matrixU(const Solver &self) { return self.matrixU(); }

  template <typename MatrixOrVector>
  static MatrixOrVector solve(const Solver &self, const MatrixOrVector &vec) {
    return self.solve(vec);
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decomposition_sparse_simplicial_cholesky_hpp__
