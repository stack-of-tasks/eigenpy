/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_decompositions_sparse_qr_hpp__
#define __eigenpy_decompositions_sparse_qr_hpp__

#include <Eigen/SparseQR>
#include <Eigen/Core>

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/decompositions/sparse/SparseSolverBase.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename SparseQRType>
struct SparseQRMatrixQTransposeReturnTypeVisitor
    : public boost::python::def_visitor<
          SparseQRMatrixQTransposeReturnTypeVisitor<SparseQRType>> {
  typedef typename SparseQRType::Scalar Scalar;
  typedef Eigen::SparseQRMatrixQTransposeReturnType<SparseQRType>
      QTransposeType;
  typedef Eigen::VectorXd VectorXd;
  typedef Eigen::MatrixXd MatrixXd;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(bp::init<const SparseQRType&>(bp::args("self", "qr")))
        .def(
            "__matmul__",
            +[](QTransposeType& self, const MatrixXd& other) -> MatrixXd {
              return MatrixXd(self * other);
            },
            bp::args("self", "other"))

        .def(
            "__matmul__",
            +[](QTransposeType& self, const VectorXd& other) -> VectorXd {
              return VectorXd(self * other);
            },
            bp::args("self", "other"));
  }

  static void expose() {
    static const std::string classname = "SparseQRMatrixQTransposeReturnType_" +
                                         scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string& name) {
    bp::class_<QTransposeType>(
        name.c_str(), "Eigen SparseQRMatrixQTransposeReturnType", bp::no_init)
        .def(SparseQRMatrixQTransposeReturnTypeVisitor())
        .def(IdVisitor<QTransposeType>());
  }
};

template <typename SparseQRType>
struct SparseQRMatrixQReturnTypeVisitor
    : public boost::python::def_visitor<
          SparseQRMatrixQReturnTypeVisitor<SparseQRType>> {
  typedef typename SparseQRType::Scalar Scalar;
  typedef Eigen::SparseQRMatrixQTransposeReturnType<SparseQRType>
      QTransposeType;
  typedef Eigen::SparseQRMatrixQReturnType<SparseQRType> QType;
  typedef typename SparseQRType::QRMatrixType QRMatrixType;
  typedef Eigen::VectorXd VectorXd;
  typedef Eigen::MatrixXd MatrixXd;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(bp::init<const SparseQRType&>(bp::args("self", "qr")))
        .def(
            "__matmul__",
            +[](QType& self, const MatrixXd& other) -> MatrixXd {
              return MatrixXd(self * other);
            },
            bp::args("self", "other"))

        .def(
            "__matmul__",
            +[](QType& self, const VectorXd& other) -> VectorXd {
              return VectorXd(self * other);
            },
            bp::args("self", "other"))

        .def("rows", &QType::rows)
        .def("cols", &QType::cols)

        .def(
            "adjoint",
            +[](const QType& self) -> QTransposeType { return self.adjoint(); })

        .def(
            "transpose",
            +[](const QType& self) -> QTransposeType {
              return self.transpose();
            })

        .def(
            "toSparse",
            +[](QType& self) -> QRMatrixType {
              Eigen::Index m = self.rows();
              MatrixXd I = MatrixXd::Identity(m, m);
              MatrixXd Q_dense = self * I;
              return Q_dense.sparseView();
            },
            bp::arg("self"),
            "Convert the Q matrix to a sparse matrix representation.");
  }

  static void expose() {
    static const std::string classname =
        "SparseQRMatrixQReturnType_" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string& name) {
    bp::class_<QType>(name.c_str(), "Eigen SparseQRMatrixQReturnType",
                      bp::no_init)
        .def(SparseQRMatrixQReturnTypeVisitor())
        .def(IdVisitor<QType>());
  }
};

template <typename SparseQRType>
struct SparseQRVisitor
    : public boost::python::def_visitor<SparseQRVisitor<SparseQRType>> {
  typedef typename SparseQRType::MatrixType MatrixType;

  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>
      DenseVectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                        MatrixType::Options>
      DenseMatrixXs;

  typedef typename SparseQRType::QRMatrixType QRMatrixType;
  typedef Eigen::SparseQRMatrixQReturnType<SparseQRType> QType;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(bp::init<>(bp::arg("self"), "Default constructor"))
        .def(bp::init<MatrixType>(
            bp::args("self", "mat"),
            "Construct a QR factorization of the matrix mat."))

        .def("cols", &SparseQRType::cols, bp::arg("self"),
             "Returns the number of columns of the represented matrix. ")
        .def("rows", &SparseQRType::rows, bp::arg("self"),
             "Returns the number of rows of the represented matrix. ")

        .def("compute", &SparseQRType::compute, bp::args("self", "matrix"),
             "Compute the symbolic and numeric factorization of the input "
             "sparse matrix. "
             "The input matrix should be in column-major storage. ")
        .def("analyzePattern", &SparseQRType::analyzePattern,
             bp::args("self", "mat"),
             "Compute the column permutation to minimize the fill-in.")
        .def("factorize", &SparseQRType::factorize, bp::args("self", "matrix"),
             "Performs a numeric decomposition of a given matrix.\n"
             "The given matrix must has the same sparcity than the matrix on "
             "which the symbolic decomposition has been performed.")

        .def("colsPermutation", &SparseQRType::colsPermutation, bp::arg("self"),
             "Returns a reference to the column matrix permutation PTc such "
             "that Pr A PTc = LU.",
             bp::return_value_policy<bp::copy_const_reference>())

        .def("info", &SparseQRType::info, bp::arg("self"),
             "NumericalIssue if the input contains INF or NaN values or "
             "overflow occured. Returns Success otherwise.")
        .def("lastErrorMessage", &SparseQRType::lastErrorMessage,
             bp::arg("self"), "Returns a string describing the type of error. ")

        .def("rank", &SparseQRType::rank, bp::arg("self"),
             "Returns the number of non linearly dependent columns as "
             "determined "
             "by the pivoting threshold. ")

        .def(
            "matrixQ",
            +[](const SparseQRType& self) -> QType { return self.matrixQ(); },
            "Returns an expression of the matrix Q as products of sparse "
            "Householder reflectors.")
        .def(
            "matrixR",
            +[](const SparseQRType& self) -> const QRMatrixType& {
              return self.matrixR();
            },
            "Returns a const reference to the \b sparse upper triangular "
            "matrix "
            "R of the QR factorization.",
            bp::return_value_policy<bp::copy_const_reference>())

        .def("setPivotThreshold", &SparseQRType::setPivotThreshold,
             bp::args("self", "thresh"),
             "Set the threshold used for a diagonal entry to be an acceptable "
             "pivot.")

        .def(SparseSolverBaseVisitor<SparseQRType>());
  }

  static void expose() {
    static const std::string classname =
        "SparseQR_" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string& name) {
    bp::class_<SparseQRType, boost::noncopyable>(
        name.c_str(),
        "Sparse left-looking QR factorization with numerical column pivoting. "
        "This class implements a left-looking QR decomposition of sparse "
        "matrices "
        "with numerical column pivoting. When a column has a norm less than a "
        "given "
        "tolerance it is implicitly permuted to the end. The QR factorization "
        "thus "
        "obtained is given by A*P = Q*R where R is upper triangular or "
        "trapezoidal. \n\n"
        "P is the column permutation which is the product of the fill-reducing "
        "and the "
        "numerical permutations. \n\n"
        "Q is the orthogonal matrix represented as products of Householder "
        "reflectors. \n\n"
        "R is the sparse triangular or trapezoidal matrix. The later occurs "
        "when A is rank-deficient. \n\n",
        bp::no_init)
        .def(SparseQRVisitor())
        .def(IdVisitor<SparseQRType>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_sparse_qr_hpp__
