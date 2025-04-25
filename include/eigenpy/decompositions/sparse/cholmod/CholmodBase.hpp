/*
 * Copyright 2024 INRIA
 */

#ifndef __eigenpy_decomposition_sparse_cholmod_cholmod_base_hpp__
#define __eigenpy_decomposition_sparse_cholmod_cholmod_base_hpp__

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/eigen/EigenBase.hpp"
#include "eigenpy/decompositions/sparse/SparseSolverBase.hpp"

#include <Eigen/CholmodSupport>

namespace eigenpy {

template <typename CholdmodDerived>
struct CholmodBaseVisitor
    : public boost::python::def_visitor<CholmodBaseVisitor<CholdmodDerived>> {
  typedef CholdmodDerived Solver;

  typedef typename CholdmodDerived::MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef MatrixType CholMatrixType;
  typedef typename MatrixType::StorageIndex StorageIndex;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def("analyzePattern", &Solver::analyzePattern,
           bp::args("self", "matrix"),
           "Performs a symbolic decomposition on the sparcity of matrix.\n"
           "This function is particularly useful when solving for several "
           "problems having the same structure.")

        .def(EigenBaseVisitor<Solver>())
        .def(SparseSolverBaseVisitor<Solver>())

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

        .def("logDeterminant", &Solver::logDeterminant, bp::arg("self"),
             "Returns the log determinant of the underlying matrix from the "
             "current factorization.")

        .def("setShift", &Solver::setShift, (bp::args("self", "offset")),
             "Sets the shift parameters that will be used to adjust the "
             "diagonal coefficients during the numerical factorization.\n"
             "During the numerical factorization, the diagonal coefficients "
             "are transformed by the following linear model: d_ii = offset + "
             "d_ii.\n"
             "The default is the identity transformation with offset=0.",
             bp::return_self<>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decomposition_sparse_cholmod_cholmod_base_hpp__
