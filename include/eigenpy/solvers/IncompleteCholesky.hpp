/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_solvers_incomplete_cholesky_hpp__
#define __eigenpy_solvers_incomplete_cholesky_hpp__

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename _MatrixType>
struct IncompleteCholeskyVisitor : public boost::python::def_visitor<
                                       IncompleteCholeskyVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Eigen::IncompleteCholesky<Scalar> Solver;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>
      DenseVectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                        MatrixType::Options>
      DenseMatrixXs;

  typedef Eigen::SparseMatrix<Scalar, Eigen::ColMajor> FactorType;
  typedef Eigen::Matrix<RealScalar, Eigen::Dynamic, 1> VectorRx;
  typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>
      PermutationType;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(bp::init<>(bp::arg("self"), "Default constructor"))
        .def(bp::init<MatrixType>(bp::args("self", "matrix"),
                                  "Constructs and performs the LDLT "
                                  "factorization from a given matrix."))

        .def("rows", &Solver::rows, bp::arg("self"),
             "Returns the number of rows of the matrix.")
        .def("cols", &Solver::cols, bp::arg("self"),
             "Returns the number of cols of the matrix.")

        .def("info", &Solver::info, bp::arg("self"),
             "Reports whether previous computation was successful.")

        .def("setInitialShift", &Solver::setInitialShift,
             bp::args("self", "shift"), "Set the initial shift parameter.")

        .def(
            "analyzePattern",
            +[](Solver& self, const MatrixType& amat) {
              self.analyzePattern(amat);
            },
            bp::arg("matrix"))
        .def(
            "factorize",
            +[](Solver& self, const MatrixType& amat) { self.factorize(amat); },
            bp::arg("matrix"))
        .def(
            "compute",
            +[](Solver& self, const MatrixType& amat) { self.compute(amat); },
            bp::arg("matrix"))

        .def(
            "matrixL",
            +[](const Solver& self) -> const FactorType& {
              return self.matrixL();
            },
            bp::return_value_policy<bp::copy_const_reference>())
        .def(
            "scalingS",
            +[](const Solver& self) -> const VectorRx& {
              return self.scalingS();
            },
            bp::return_value_policy<bp::copy_const_reference>())
        .def(
            "permutationP",
            +[](const Solver& self) -> const PermutationType& {
              return self.permutationP();
            },
            bp::return_value_policy<bp::copy_const_reference>())

        .def(
            "solve",
            +[](const Solver& self, const Eigen::Ref<DenseVectorXs const>& b)
                -> DenseVectorXs { return self.solve(b); },
            bp::arg("b"),
            "Returns the solution x of A x = b using the current decomposition "
            "of A, where b is a right hand side vector.")
        .def(
            "solve",
            +[](const Solver& self, const Eigen::Ref<DenseMatrixXs const>& B)
                -> DenseMatrixXs { return self.solve(B); },
            bp::arg("b"),
            "Returns the solution X of A X = B using the current decomposition "
            "of A where B is a right hand side matrix.")
        .def(
            "solve",
            +[](const Solver& self, const MatrixType& B) -> MatrixType {
              DenseMatrixXs B_dense = DenseMatrixXs(B);
              DenseMatrixXs X_dense = self.solve(B_dense);
              return MatrixType(X_dense.sparseView());
            },
            bp::arg("b"),
            "Returns the solution X of A X = B using the current decomposition "
            "of A where B is a right hand side matrix.");
  }

  static void expose() {
    static const std::string classname =
        "IncompleteCholesky_" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string& name) {
    bp::class_<Solver, boost::noncopyable>(name.c_str(), "Incomplete Cholesky.",
                                           bp::no_init)
        .def(IncompleteCholeskyVisitor())
        .def(IdVisitor<Solver>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_solvers_incomplete_cholesky_hpp__
