/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_decompositions_partialpivlu_hpp__
#define __eigenpy_decompositions_partialpivlu_hpp__

#include <Eigen/LU>
#include <Eigen/Core>

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"
#include "eigenpy/eigen/EigenBase.hpp"

namespace eigenpy {

template <typename _MatrixType>
struct PartialPivLUSolverVisitor : public boost::python::def_visitor<
                                       PartialPivLUSolverVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>
      VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                        MatrixType::Options>
      MatrixXs;
  typedef Eigen::PartialPivLU<MatrixType> Solver;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(bp::init<>(bp::arg("self"), "Default constructor"))
        .def(bp::init<Eigen::DenseIndex>(
            bp::args("self", "size"),
            "Default constructor with memory preallocation"))
        .def(bp::init<MatrixType>(
            bp::args("self", "matrix"),
            "Constructs a LU factorization from a given matrix."))

        .def(EigenBaseVisitor<Solver>())

        .def("determinant", &Solver::determinant, bp::arg("self"),
             "Returns the determinant of the matrix of which *this is the LU "
             "decomposition.")
        .def(
            "compute",
            (Solver & (Solver::*)(const Eigen::EigenBase<MatrixType> &matrix)) &
                Solver::compute,
            bp::args("self", "matrix"),
            "Computes the LU factorization of given matrix.",
            bp::return_self<>())
        .def(
            "inverse",
            +[](const Solver &self) -> MatrixType { return self.inverse(); },
            bp::arg("self"),
            "Returns the inverse of the matrix of which *this is the LU "
            "decomposition.")
        .def("matrixLU", &Solver::matrixLU, bp::arg("self"),
             "Returns the LU decomposition matrix.",
             bp::return_internal_reference<>())

        .def("permutationP", &Solver::permutationP, bp::arg("self"),
             "Returns the permutation P.",
             bp::return_value_policy<bp::copy_const_reference>())

        .def("rcond", &Solver::rcond, bp::arg("self"),
             "Returns an estimate of the reciprocal condition number of the "
             "matrix.")
        .def("reconstructedMatrix", &Solver::reconstructedMatrix,
             bp::arg("self"),
             "Returns the matrix represented by the decomposition, i.e., it "
             "returns the product: P-1LUQ-1. This function is provided for "
             "debug "
             "purpose.")
        .def("solve", &solve<VectorXs>, bp::args("self", "b"),
             "Returns the solution x of A x = b using the current "
             "decomposition of A.")
        .def("solve", &solve<MatrixXs>, bp::args("self", "B"),
             "Returns the solution X of A X = B using the current "
             "decomposition of A where B is a right hand side matrix.");
  }

  static void expose() {
    static const std::string classname =
        "FullPivLU" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string &name) {
    bp::class_<Solver>(
        name.c_str(),
        "LU decomposition of a matrix with partial pivoting, "
        "and related features. \n\n"
        "This class represents a LU decomposition of a square "
        "invertible matrix, "
        "with partial pivoting: the matrix A is decomposed as A "
        "= PLU where L is "
        "unit-lower-triangular, U is upper-triangular, and P is "
        "a permutation matrix.\n\n"
        "Typically, partial pivoting LU decomposition is only "
        "considered numerically "
        "stable for square invertible matrices. Thus LAPACK's "
        "dgesv and dgesvx require "
        "the matrix to be square and invertible. The present "
        "class does the same. It "
        "will assert that the matrix is square, but it won't "
        "(actually it can't) check "
        "that the matrix is invertible: it is your task to "
        "check that you only use this "
        "decomposition on invertible matrices. \n\n"
        "The guaranteed safe alternative, working for all matrices, "
        "is the full pivoting LU decomposition, provided by class "
        "FullPivLU. \n\n"
        "This is not a rank-revealing LU decomposition. Many features "
        "are intentionally absent from this class, such as "
        "rank computation. If you need these features, use class "
        "FullPivLU. \n\n"
        "This LU decomposition is suitable to invert invertible "
        "matrices. It is what MatrixBase::inverse() uses in the "
        "general case. On the other hand, it is not suitable to "
        "determine whether a given matrix is invertible. \n\n"
        "The data of the LU decomposition can be directly accessed "
        "through the methods matrixLU(), permutationP().",
        bp::no_init)
        .def(IdVisitor<Solver>())
        .def(PartialPivLUSolverVisitor());
  }

 private:
  template <typename MatrixOrVector>
  static MatrixOrVector solve(const Solver &self, const MatrixOrVector &vec) {
    return self.solve(vec);
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_partialpivlu_hpp__
