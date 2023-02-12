/*
 * Copyright 2020-2021 INRIA
 */

#ifndef __eigenpy_decomposition_ldlt_hpp__
#define __eigenpy_decomposition_ldlt_hpp__

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename _MatrixType>
struct LDLTSolverVisitor
    : public boost::python::def_visitor<LDLTSolverVisitor<_MatrixType> > {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>
      VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                        MatrixType::Options>
      MatrixXs;
  typedef Eigen::LDLT<MatrixType> Solver;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(bp::init<>(bp::arg("self"), "Default constructor"))
        .def(bp::init<Eigen::DenseIndex>(
            bp::args("self", "size"),
            "Default constructor with memory preallocation"))
        .def(bp::init<MatrixType>(
            bp::args("self", "matrix"),
            "Constructs a LDLT factorization from a given matrix."))

        .def("isNegative", &Solver::isNegative, bp::arg("self"),
             "Returns true if the matrix is negative (semidefinite).")
        .def("isPositive", &Solver::isPositive, bp::arg("self"),
             "Returns true if the matrix is positive (semidefinite).")

        .def("matrixL", &matrixL, bp::arg("self"),
             "Returns the lower triangular matrix L.")
        .def("matrixU", &matrixU, bp::arg("self"),
             "Returns the upper triangular matrix U.")
        .def("vectorD", &vectorD, bp::arg("self"),
             "Returns the coefficients of the diagonal matrix D.")
        .def("transpositionsP", &transpositionsP, bp::arg("self"),
             "Returns the permutation matrix P.")

        .def("matrixLDLT", &Solver::matrixLDLT, bp::arg("self"),
             "Returns the LDLT decomposition matrix.",
             bp::return_internal_reference<>())

        .def("rankUpdate",
             (Solver & (Solver::*)(const Eigen::MatrixBase<VectorXs> &,
                                   const RealScalar &)) &
                 Solver::template rankUpdate<VectorXs>,
             bp::args("self", "vector", "sigma"), bp::return_self<>())

#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
        .def("adjoint", &Solver::adjoint, bp::arg("self"),
             "Returns the adjoint, that is, a reference to the decomposition "
             "itself as if the underlying matrix is self-adjoint.",
             bp::return_self<>())
#endif

        .def(
            "compute",
            (Solver & (Solver::*)(const Eigen::EigenBase<MatrixType> &matrix)) &
                Solver::compute,
            bp::args("self", "matrix"), "Computes the LDLT of given matrix.",
            bp::return_self<>())

        .def("info", &Solver::info, bp::arg("self"),
             "NumericalIssue if the input contains INF or NaN values or "
             "overflow occured. Returns Success otherwise.")
#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
        .def("rcond", &Solver::rcond, bp::arg("self"),
             "Returns an estimate of the reciprocal condition number of the "
             "matrix.")
#endif
        .def("reconstructedMatrix", &Solver::reconstructedMatrix,
             bp::arg("self"),
             "Returns the matrix represented by the decomposition, i.e., it "
             "returns the product: L L^*. This function is provided for debug "
             "purpose.")
        .def("solve", &solve<VectorXs>, bp::args("self", "b"),
             "Returns the solution x of A x = b using the current "
             "decomposition of A.")
        .def("solve", &solve<MatrixXs>, bp::args("self", "B"),
             "Returns the solution X of A X = B using the current "
             "decomposition of A where B is a right hand side matrix.")

        .def("setZero", &Solver::setZero, bp::arg("self"),
             "Clear any existing decomposition.");
  }

  static void expose() {
    static const std::string classname =
        "LDLT" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string &name) {
    bp::class_<Solver>(
        name.c_str(),
        "Robust Cholesky decomposition of a matrix with pivoting.\n\n"
        "Perform a robust Cholesky decomposition of a positive semidefinite or "
        "negative semidefinite matrix $ A $ such that $ A = P^TLDL^*P $, where "
        "P is a permutation matrix, L is lower triangular with a unit diagonal "
        "and D is a diagonal matrix.\n\n"
        "The decomposition uses pivoting to ensure stability, so that L will "
        "have zeros in the bottom right rank(A) - n submatrix. Avoiding the "
        "square root on D also stabilizes the computation.",
        bp::no_init)
        .def(LDLTSolverVisitor());
  }

 private:
  static MatrixType matrixL(const Solver &self) { return self.matrixL(); }
  static MatrixType matrixU(const Solver &self) { return self.matrixU(); }
  static VectorXs vectorD(const Solver &self) { return self.vectorD(); }

  static MatrixType transpositionsP(const Solver &self) {
    return self.transpositionsP() *
           MatrixType::Identity(self.matrixL().rows(), self.matrixL().rows());
  }

  template <typename MatrixOrVector>
  static MatrixOrVector solve(const Solver &self, const MatrixOrVector &vec) {
    return self.solve(vec);
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decomposition_ldlt_hpp__
