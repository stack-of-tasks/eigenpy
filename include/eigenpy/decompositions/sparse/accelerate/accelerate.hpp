/*
 * Copyright 2024 INRIA
 */

#ifndef __eigenpy_decomposition_sparse_accelerate_accelerate_hpp__
#define __eigenpy_decomposition_sparse_accelerate_accelerate_hpp__

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/eigen/EigenBase.hpp"
#include "eigenpy/decompositions/sparse/SparseSolverBase.hpp"

#include <Eigen/AccelerateSupport>

namespace eigenpy {

template <typename AccelerateDerived>
struct AccelerateImplVisitor : public boost::python::def_visitor<
                                   AccelerateImplVisitor<AccelerateDerived>> {
  typedef AccelerateDerived Solver;

  typedef typename AccelerateDerived::MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef MatrixType CholMatrixType;
  typedef typename MatrixType::StorageIndex StorageIndex;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl

        .def("analyzePattern", &Solver::analyzePattern,
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

        .def("factorize", &Solver::factorize, bp::args("self", "matrix"),
             "Performs a numeric decomposition of a given matrix.\n"
             "The given matrix must has the same sparcity than the matrix on "
             "which the symbolic decomposition has been performed.\n"
             "See also analyzePattern().")

        .def("info", &Solver::info, bp::arg("self"),
             "NumericalIssue if the input contains INF or NaN values or "
             "overflow occured. Returns Success otherwise.")

        .def("setOrder", &Solver::setOrder, bp::arg("self"), "Set order");
  }

  static void expose(const std::string &name, const std::string &doc = "") {
    bp::class_<Solver, boost::noncopyable>(name.c_str(), doc.c_str(),
                                           bp::no_init)
        .def(AccelerateImplVisitor())

        .def(bp::init<>(bp::arg("self"), "Default constructor"))
        .def(bp::init<MatrixType>(bp::args("self", "matrix"),
                                  "Constructs and performs the "
                                  "factorization from a given matrix."));
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decomposition_sparse_accelerate_accelerate_hpp__
