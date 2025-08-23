/*
 * Copyright 2024 INRIA
 */

#ifndef __eigenpy_decompositions_houselholder_qr_hpp__
#define __eigenpy_decompositions_houselholder_qr_hpp__

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"

#include <Eigen/QR>

namespace eigenpy {

template <typename _MatrixType>
struct HouseholderQRSolverVisitor
    : public boost::python::def_visitor<
          HouseholderQRSolverVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>
      VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                        MatrixType::Options>
      MatrixXs;
  typedef Eigen::HouseholderQR<MatrixType> Solver;
  typedef Solver Self;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(bp::init<>(bp::arg("self"),
                      "Default constructor.\n"
                      "The default constructor is useful in cases in which the "
                      "user intends to perform decompositions via "
                      "HouseholderQR.compute(matrix)"))
        .def(bp::init<Eigen::DenseIndex, Eigen::DenseIndex>(
            bp::args("self", "rows", "cols"),
            "Default constructor with memory preallocation.\n"
            "Like the default constructor but with preallocation of the "
            "internal data according to the specified problem size. "))
        .def(bp::init<MatrixType>(
            bp::args("self", "matrix"),
            "Constructs a QR factorization from a given matrix.\n"
            "This constructor computes the QR factorization of the matrix "
            "matrix by calling the method compute()."))

        .def("absDeterminant", &Self::absDeterminant, bp::arg("self"),
             "Returns the absolute value of the determinant of the matrix of "
             "which *this is the QR decomposition.\n"
             "It has only linear complexity (that is, O(n) where n is the "
             "dimension of the square matrix) as the QR decomposition has "
             "already been computed.\n"
             "Note: This is only for square matrices.")
        .def("logAbsDeterminant", &Self::logAbsDeterminant, bp::arg("self"),
             "Returns the natural log of the absolute value of the determinant "
             "of the matrix of which *this is the QR decomposition.\n"
             "It has only linear complexity (that is, O(n) where n is the "
             "dimension of the square matrix) as the QR decomposition has "
             "already been computed.\n"
             "Note: This is only for square matrices. This method is useful to "
             "work around the risk of overflow/underflow that's inherent to "
             "determinant computation.")

        .def("matrixQR", &Self::matrixQR, bp::arg("self"),
             "Returns the matrix where the Householder QR decomposition is "
             "stored in a LAPACK-compatible way.",
             bp::return_value_policy<bp::copy_const_reference>())

        .def(
            "compute",
            (Solver & (Solver::*)(const Eigen::EigenBase<MatrixType> &matrix)) &
                Solver::compute,
            bp::args("self", "matrix"),
            "Computes the QR factorization of given matrix.",
            bp::return_self<>())

        .def("solve", &solve<MatrixXs>, bp::args("self", "B"),
             "Returns the solution X of A X = B using the current "
             "decomposition of A where B is a right hand side matrix.");
  }

  static void expose() {
    static const std::string classname =
        "HouseholderQR" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string &name) {
    bp::class_<Solver>(
        name.c_str(),
        "This class performs a QR decomposition of a matrix A into matrices Q "
        "and R such that A=QR by using Householder transformations.\n"
        "Here, Q a unitary matrix and R an upper triangular matrix. The result "
        "is stored in a compact way compatible with LAPACK.\n"
        "\n"
        "Note that no pivoting is performed. This is not a rank-revealing "
        "decomposition. If you want that feature, use FullPivHouseholderQR or "
        "ColPivHouseholderQR instead.\n"
        "\n"
        "This Householder QR decomposition is faster, but less numerically "
        "stable and less feature-full than FullPivHouseholderQR or "
        "ColPivHouseholderQR.",
        bp::no_init)
        .def(HouseholderQRSolverVisitor())
        .def(IdVisitor<Solver>());
  }

 private:
  template <typename MatrixOrVector>
  static MatrixOrVector solve(const Solver &self, const MatrixOrVector &vec) {
    return self.solve(vec);
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_houselholder_qr_hpp__
