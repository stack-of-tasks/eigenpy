/*
 * Copyright 2024 INRIA
 */

#ifndef __eigenpy_decompositions_col_piv_houselholder_qr_hpp__
#define __eigenpy_decompositions_col_piv_houselholder_qr_hpp__

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"

#include <Eigen/QR>

namespace eigenpy {

template <typename _MatrixType>
struct ColPivHouseholderQRSolverVisitor
    : public boost::python::def_visitor<
          ColPivHouseholderQRSolverVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>
      VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                        MatrixType::Options>
      MatrixXs;
  typedef Eigen::ColPivHouseholderQR<MatrixType> Solver;
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
        .def("dimensionOfKernel", &Self::dimensionOfKernel, bp::arg("self"),
             "Returns the dimension of the kernel of the matrix of which *this "
             "is the QR decomposition.")
        .def("info", &Self::info, bp::arg("self"),
             "Reports whether the QR factorization was successful.\n"
             "Note: This function always returns Success. It is provided for "
             "compatibility with other factorization routines.")
        .def("isInjective", &Self::isInjective, bp::arg("self"),
             "Returns true if the matrix associated with this QR decomposition "
             "represents an injective linear map, i.e. has trivial kernel; "
             "false otherwise.\n"
             "\n"
             "Note: This method has to determine which pivots should be "
             "considered nonzero. For that, it uses the threshold value that "
             "you can control by calling setThreshold(threshold).")
        .def("isInvertible", &Self::isInvertible, bp::arg("self"),
             "Returns true if the matrix associated with the QR decomposition "
             "is invertible.\n"
             "\n"
             "Note: This method has to determine which pivots should be "
             "considered nonzero. For that, it uses the threshold value that "
             "you can control by calling setThreshold(threshold).")
        .def("isSurjective", &Self::isSurjective, bp::arg("self"),
             "Returns true if the matrix associated with this QR decomposition "
             "represents a surjective linear map; false otherwise.\n"
             "\n"
             "Note: This method has to determine which pivots should be "
             "considered nonzero. For that, it uses the threshold value that "
             "you can control by calling setThreshold(threshold).")
        .def("maxPivot", &Self::maxPivot, bp::arg("self"),
             "Returns the absolute value of the biggest pivot, i.e. the "
             "biggest diagonal coefficient of U.")
        .def("nonzeroPivots", &Self::nonzeroPivots, bp::arg("self"),
             "Returns the number of nonzero pivots in the QR decomposition. "
             "Here nonzero is meant in the exact sense, not in a fuzzy sense. "
             "So that notion isn't really intrinsically interesting, but it is "
             "still useful when implementing algorithms.")
        .def("rank", &Self::rank, bp::arg("self"),
             "Returns the rank of the matrix associated with the QR "
             "decomposition.\n"
             "\n"
             "Note: This method has to determine which pivots should be "
             "considered nonzero. For that, it uses the threshold value that "
             "you can control by calling setThreshold(threshold).")

        .def("setThreshold",
             (Self & (Self::*)(const RealScalar &)) & Self::setThreshold,
             bp::args("self", "threshold"),
             "Allows to prescribe a threshold to be used by certain methods, "
             "such as rank(), who need to determine when pivots are to be "
             "considered nonzero. This is not used for the QR decomposition "
             "itself.\n"
             "\n"
             "When it needs to get the threshold value, Eigen calls "
             "threshold(). By default, this uses a formula to automatically "
             "determine a reasonable threshold. Once you have called the "
             "present method setThreshold(const RealScalar&), your value is "
             "used instead.\n"
             "\n"
             "Note: A pivot will be considered nonzero if its absolute value "
             "is strictly greater than |pivot| ⩽ threshold×|maxpivot| where "
             "maxpivot is the biggest pivot.",
             bp::return_self<>())
        .def("threshold", &Self::threshold, bp::arg("self"),
             "Returns the threshold that will be used by certain methods such "
             "as rank().")

        .def("matrixQR", &Self::matrixQR, bp::arg("self"),
             "Returns the matrix where the Householder QR decomposition is "
             "stored in a LAPACK-compatible way.",
             bp::return_value_policy<bp::copy_const_reference>())
        .def("matrixR", &Self::matrixR, bp::arg("self"),
             "Returns the matrix where the result Householder QR is stored.",
             bp::return_value_policy<bp::copy_const_reference>())

        .def(
            "compute",
            (Solver & (Solver::*)(const Eigen::EigenBase<MatrixType> &matrix)) &
                Solver::compute,
            bp::args("self", "matrix"),
            "Computes the QR factorization of given matrix.",
            bp::return_self<>())

        .def("inverse", inverse, bp::arg("self"),
             "Returns the inverse of the matrix associated with the QR "
             "decomposition..")

        .def("solve", &solve<MatrixXs>, bp::args("self", "B"),
             "Returns the solution X of A X = B using the current "
             "decomposition of A where B is a right hand side matrix.");
  }

  static void expose() {
    static const std::string classname =
        "ColPivHouseholderQR" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string &name) {
    bp::class_<Solver>(
        name.c_str(),
        "This class performs a rank-revealing QR decomposition of a matrix A "
        "into matrices P, Q and R such that:\n"
        "AP=QR\n"
        "by using Householder transformations. Here, P is a permutation "
        "matrix, Q a unitary matrix and R an upper triangular matrix.\n"
        "\n"
        "This decomposition performs column pivoting in order to be "
        "rank-revealing and improve numerical stability. It is slower than "
        "HouseholderQR, and faster than FullPivHouseholderQR.",
        bp::no_init)
        .def(ColPivHouseholderQRSolverVisitor())
        .def(IdVisitor<Solver>());
  }

 private:
  template <typename MatrixOrVector>
  static MatrixOrVector solve(const Solver &self, const MatrixOrVector &vec) {
    return self.solve(vec);
  }
  static MatrixXs inverse(const Self &self) { return self.inverse(); }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_col_piv_houselholder_qr_hpp__
