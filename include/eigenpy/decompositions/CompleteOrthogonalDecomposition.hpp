/*
 * Copyright 2024 INRIA
 */

#ifndef __eigenpy_decompositions_complete_orthogonal_decomposition_hpp__
#define __eigenpy_decompositions_complete_orthogonal_decomposition_hpp__

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"

#include <Eigen/QR>

namespace eigenpy {

template <typename _MatrixType>
struct CompleteOrthogonalDecompositionSolverVisitor
    : public boost::python::def_visitor<
          CompleteOrthogonalDecompositionSolverVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>
      VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                        MatrixType::Options>
      MatrixXs;
  typedef Eigen::CompleteOrthogonalDecomposition<MatrixType> Solver;
  typedef Solver Self;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(bp::init<>(bp::arg("self"),
                      "Default constructor.\n"
                      "The default constructor is useful in cases in which the "
                      "user intends to perform decompositions via "
                      "CompleteOrthogonalDecomposition.compute(matrix)"))
        .def(bp::init<Eigen::DenseIndex, Eigen::DenseIndex>(
            bp::args("self", "rows", "cols"),
            "Default constructor with memory preallocation.\n"
            "Like the default constructor but with preallocation of the "
            "internal data according to the specified problem size. "))
        .def(bp::init<MatrixType>(bp::args("self", "matrix"),
                                  "Constructs a complete orthogonal "
                                  "factorization from a given matrix.\n"
                                  "This constructor computes the complete "
                                  "orthogonal factorization of the matrix "
                                  "matrix by calling the method compute()."))

        .def("absDeterminant", &Self::absDeterminant, bp::arg("self"),
             "Returns the absolute value of the determinant of the matrix "
             "associated with the complete orthogonal decomposition.\n"
             "It has only linear complexity (that is, O(n) where n is the "
             "dimension of the square matrix) as the complete orthogonal "
             "decomposition has "
             "already been computed.\n"
             "Note: This is only for square matrices.")
        .def("logAbsDeterminant", &Self::logAbsDeterminant, bp::arg("self"),
             "Returns the natural log of the absolute value of the determinant "
             "of the matrix of which *this is the complete orthogonal "
             "decomposition.\n"
             "It has only linear complexity (that is, O(n) where n is the "
             "dimension of the square matrix) as the complete orthogonal "
             "decomposition has "
             "already been computed.\n"
             "Note: This is only for square matrices. This method is useful to "
             "work around the risk of overflow/underflow that's inherent to "
             "determinant computation.")
        .def("dimensionOfKernel", &Self::dimensionOfKernel, bp::arg("self"),
             "Returns the dimension of the kernel of the matrix of which *this "
             "is the complete orthogonal decomposition.")
        .def("info", &Self::info, bp::arg("self"),
             "Reports whether the complete orthogonal factorization was "
             "successful.\n"
             "Note: This function always returns Success. It is provided for "
             "compatibility with other factorization routines.")
        .def("isInjective", &Self::isInjective, bp::arg("self"),
             "Returns true if the matrix associated with this complete "
             "orthogonal decomposition "
             "represents an injective linear map, i.e. has trivial kernel; "
             "false otherwise.\n"
             "\n"
             "Note: This method has to determine which pivots should be "
             "considered nonzero. For that, it uses the threshold value that "
             "you can control by calling setThreshold(threshold).")
        .def("isInvertible", &Self::isInvertible, bp::arg("self"),
             "Returns true if the matrix associated with the complete "
             "orthogonal decomposition "
             "is invertible.\n"
             "\n"
             "Note: This method has to determine which pivots should be "
             "considered nonzero. For that, it uses the threshold value that "
             "you can control by calling setThreshold(threshold).")
        .def("isSurjective", &Self::isSurjective, bp::arg("self"),
             "Returns true if the matrix associated with this complete "
             "orthogonal decomposition "
             "represents a surjective linear map; false otherwise.\n"
             "\n"
             "Note: This method has to determine which pivots should be "
             "considered nonzero. For that, it uses the threshold value that "
             "you can control by calling setThreshold(threshold).")
        .def("maxPivot", &Self::maxPivot, bp::arg("self"),
             "Returns the absolute value of the biggest pivot, i.e. the "
             "biggest diagonal coefficient of U.")
        .def("nonzeroPivots", &Self::nonzeroPivots, bp::arg("self"),
             "Returns the number of nonzero pivots in the complete orthogonal "
             "decomposition. "
             "Here nonzero is meant in the exact sense, not in a fuzzy sense. "
             "So that notion isn't really intrinsically interesting, but it is "
             "still useful when implementing algorithms.")
        .def("rank", &Self::rank, bp::arg("self"),
             "Returns the rank of the matrix associated with the complete "
             "orthogonal "
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
             "considered nonzero. This is not used for the complete orthogonal "
             "decomposition "
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

        .def("matrixQTZ", &Self::matrixQTZ, bp::arg("self"),
             "Returns the matrix where the complete orthogonal decomposition "
             "is stored.",
             bp::return_value_policy<bp::copy_const_reference>())
        .def("matrixT", &Self::matrixT, bp::arg("self"),
             "Returns the matrix where the complete orthogonal decomposition "
             "is stored.",
             bp::return_value_policy<bp::copy_const_reference>())
        .def("matrixZ", &Self::matrixZ, bp::arg("self"),
             "Returns the matrix Z.")

        .def(
            "compute",
            (Solver & (Solver::*)(const Eigen::EigenBase<MatrixType> &matrix)) &
                Solver::compute,
            bp::args("self", "matrix"),
            "Computes the complete orthogonal factorization of given matrix.",
            bp::return_self<>())

        .def("pseudoInverse", pseudoInverse, bp::arg("self"),
             "Returns the pseudo-inverse of the matrix associated with the "
             "complete orthogonal "
             "decomposition.")

        .def("solve", &solve<MatrixXs>, bp::args("self", "B"),
             "Returns the solution X of A X = B using the current "
             "decomposition of A where B is a right hand side matrix.");
  }

  static void expose() {
    static const std::string classname =
        "CompleteOrthogonalDecomposition" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string &name) {
    bp::class_<Solver>(
        name.c_str(),
        "This class performs a rank-revealing complete orthogonal "
        "decomposition of a matrix A into matrices P, Q, T, and Z such that:\n"
        "AP=Q[T000]Z"
        "by using Householder transformations. Here, P is a permutation "
        "matrix, Q and Z are unitary matrices and T an upper triangular matrix "
        "of size rank-by-rank. A may be rank deficient.",
        bp::no_init)
        .def(CompleteOrthogonalDecompositionSolverVisitor())
        .def(IdVisitor<Solver>());
  }

 private:
  template <typename MatrixOrVector>
  static MatrixOrVector solve(const Solver &self, const MatrixOrVector &vec) {
    return self.solve(vec);
  }
  static MatrixXs pseudoInverse(const Self &self) {
    return self.pseudoInverse();
  }
};

}  // namespace eigenpy

#endif  // ifndef
        // __eigenpy_decompositions_complete_orthogonal_decomposition_hpp__
