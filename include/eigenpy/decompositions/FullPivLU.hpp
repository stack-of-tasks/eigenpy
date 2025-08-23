/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_decompositions_fullpivlu_hpp__
#define __eigenpy_decompositions_fullpivlu_hpp__

#include <Eigen/LU>
#include <Eigen/Core>

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"
#include "eigenpy/eigen/EigenBase.hpp"

namespace eigenpy {

template <typename _MatrixType>
struct FullPivLUSolverVisitor
    : public boost::python::def_visitor<FullPivLUSolverVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>
      VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                        MatrixType::Options>
      MatrixXs;
  typedef Eigen::FullPivLU<MatrixType> Solver;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(bp::init<>(bp::arg("self"), "Default constructor"))
        .def(bp::init<Eigen::DenseIndex, Eigen::DenseIndex>(
            bp::args("self", "rows", "cols"),
            "Default constructor with memory preallocation"))
        .def(bp::init<MatrixType>(
            bp::args("self", "matrix"),
            "Constructs a LU factorization from a given matrix."))

        .def(EigenBaseVisitor<Solver>())

        .def(
            "compute",
            (Solver & (Solver::*)(const Eigen::EigenBase<MatrixType> &matrix)) &
                Solver::compute,
            bp::args("self", "matrix"),
            "Computes the LU decomposition of the given matrix.",
            bp::return_self<>())

        .def("determinant", &Solver::determinant, bp::arg("self"),
             "Returns the determinant of the matrix of which *this is the LU "
             "decomposition.")
        .def("dimensionOfKernel", &Solver::dimensionOfKernel, bp::arg("self"),
             "Returns the dimension of the kernel of the matrix of which *this "
             "is the LU decomposition.")
        .def(
            "image",
            +[](Solver &self, const MatrixType &mat) -> MatrixType {
              return self.image(mat);
            },
            bp::args("self", "originalMatrix"),
            "Returns the image of the matrix, also called its column-space. "
            "The columns of the returned matrix will form a basis of the "
            "image (column-space).")
        .def(
            "inverse",
            +[](Solver &self) -> MatrixType { return self.inverse(); },
            bp::arg("self"),
            "Returns the inverse of the matrix of which *this is the LU "
            "decomposition.")

        .def("isInjective", &Solver::isInjective, bp::arg("self"))
        .def("isInvertible", &Solver::isInvertible, bp::arg("self"))
        .def("isSurjective", &Solver::isSurjective, bp::arg("self"))

        .def(
            "kernel", +[](Solver &self) -> MatrixType { return self.kernel(); },
            bp::arg("self"),
            "Returns the kernel of the matrix, also called its null-space. "
            "The columns of the returned matrix will form a basis of the "
            "kernel.")

        .def("matrixLU", &Solver::matrixLU, bp::arg("self"),
             "Returns the LU decomposition matrix.",
             bp::return_internal_reference<>())

        .def("maxPivot", &Solver::maxPivot, bp::arg("self"))
        .def("nonzeroPivots", &Solver::nonzeroPivots, bp::arg("self"))

        .def("permutationP", &Solver::permutationP, bp::arg("self"),
             "Returns the permutation P.",
             bp::return_value_policy<bp::copy_const_reference>())
        .def("permutationQ", &Solver::permutationQ, bp::arg("self"),
             "Returns the permutation Q.",
             bp::return_value_policy<bp::copy_const_reference>())

        .def("rank", &Solver::rank, bp::arg("self"))

        .def("rcond", &Solver::rcond, bp::arg("self"),
             "Returns an estimate of the reciprocal condition number of the "
             "matrix.")
        .def("reconstructedMatrix", &Solver::reconstructedMatrix,
             bp::arg("self"),
             "Returns the matrix represented by the decomposition, i.e., it "
             "returns the product: P-1LUQ-1. This function is provided for "
             "debug "
             "purpose.")

        .def("setThreshold",
             (Solver & (Solver::*)(const RealScalar &)) & Solver::setThreshold,
             bp::args("self", "threshold"),
             "Allows to prescribe a threshold to be used by certain methods, "
             "such as rank(), who need to determine when pivots are to be "
             "considered nonzero. This is not used for the LU decomposition "
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
        .def(
            "setThreshold",
            +[](Solver &self) -> Solver & {
              return self.setThreshold(Eigen::Default);
            },
            bp::arg("self"),
            "Allows to come back to the default behavior, letting Eigen use "
            "its default formula for determining the threshold.",
            bp::return_self<>())

        .def("solve", &solve<VectorXs>, bp::args("self", "b"),
             "Returns the solution x of A x = b using the current "
             "decomposition of A.")
        .def("solve", &solve<MatrixXs>, bp::args("self", "B"),
             "Returns the solution X of A X = B using the current "
             "decomposition of A where B is a right hand side matrix.")

        .def("threshold", &Solver::threshold, bp::arg("self"),
             "Returns the threshold that will be used by certain methods such "
             "as rank().");
  }

  static void expose() {
    static const std::string classname =
        "FullPivLU" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string &name) {
    bp::class_<Solver>(
        name.c_str(),
        "LU decomposition of a matrix with complete pivoting, and related "
        "features.\n\n"
        "This class represents a LU decomposition of any matrix, with complete "
        "pivoting: the matrix A is decomposed as A=P−1LUQ−1 where L is "
        "unit-lower-triangular, U is upper-triangular, and P and Q are "
        "permutation matrices. This is a rank-revealing LU decomposition. "
        "The eigenvalues (diagonal coefficients) of U are sorted in such a "
        "way that any zeros are at the end.\n\n"
        "This decomposition provides the generic approach to solving systems "
        "of "
        "linear equations, computing the rank, invertibility, inverse, kernel, "
        "and determinant. \n\n"
        "This LU decomposition is very stable and well tested with large "
        "matrices. "
        "However there are use cases where the SVD decomposition is inherently "
        "more "
        "stable and/or flexible. For example, when computing the kernel of a "
        "matrix, "
        "working with the SVD allows to select the smallest singular values of "
        "the matrix, something that the LU decomposition doesn't see.",
        bp::no_init)
        .def(IdVisitor<Solver>())
        .def(FullPivLUSolverVisitor());
  }

 private:
  template <typename MatrixOrVector>
  static MatrixOrVector solve(const Solver &self, const MatrixOrVector &vec) {
    return self.solve(vec);
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_fullpivlu_hpp__
