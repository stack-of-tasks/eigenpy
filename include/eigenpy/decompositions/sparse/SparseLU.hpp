/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_decompositions_sparse_lu_hpp__
#define __eigenpy_decompositions_sparse_lu_hpp__

#include <Eigen/SparseLU>
#include <Eigen/Core>

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/decompositions/sparse/SparseSolverBase.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename MappedSupernodalType>
struct SparseLUMatrixLReturnTypeVisitor
    : public boost::python::def_visitor<
          SparseLUMatrixLReturnTypeVisitor<MappedSupernodalType>> {
  typedef Eigen::SparseLUMatrixLReturnType<MappedSupernodalType> LType;
  typedef typename MappedSupernodalType::Scalar Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
      MatrixXs;

  template <typename MatrixOrVector>
  static void solveInPlace(const LType &self,
                           Eigen::Ref<MatrixOrVector> mat_vec) {
    self.solveInPlace(mat_vec);
  }

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(bp::init<MappedSupernodalType>(bp::args("self", "mapL")))

        .def("rows", &LType::rows)
        .def("cols", &LType::cols)

        .def("solveInPlace", &solveInPlace<MatrixXs>, bp::args("self", "X"))
        .def("solveInPlace", &solveInPlace<VectorXs>, bp::args("self", "x"));
  }

  static void expose(const std::string &name) {
    bp::class_<LType>(name.c_str(), "Eigen SparseLUMatrixLReturnType",
                      bp::no_init)
        .def(SparseLUMatrixLReturnTypeVisitor())
        .def(IdVisitor<LType>());
  }
};

template <typename MatrixLType, typename MatrixUType>
struct SparseLUMatrixUReturnTypeVisitor
    : public boost::python::def_visitor<
          SparseLUMatrixUReturnTypeVisitor<MatrixLType, MatrixUType>> {
  typedef Eigen::SparseLUMatrixUReturnType<MatrixLType, MatrixUType> UType;
  typedef typename MatrixLType::Scalar Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
      MatrixXs;

  template <typename MatrixOrVector>
  static void solveInPlace(const UType &self,
                           Eigen::Ref<MatrixOrVector> mat_vec) {
    self.solveInPlace(mat_vec);
  }

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(bp::init<MatrixLType, MatrixUType>(bp::args("self", "mapL", "mapU")))

        .def("rows", &UType::rows)
        .def("cols", &UType::cols)

        .def("solveInPlace", &solveInPlace<MatrixXs>, bp::args("self", "X"))
        .def("solveInPlace", &solveInPlace<VectorXs>, bp::args("self", "x"));
  }

  static void expose(const std::string &name) {
    bp::class_<UType>(name.c_str(), "Eigen SparseLUMatrixUReturnType",
                      bp::no_init)
        .def(SparseLUMatrixUReturnTypeVisitor())
        .def(IdVisitor<UType>());
  }
};

template <typename _MatrixType,
          typename _Ordering =
              Eigen::COLAMDOrdering<typename _MatrixType::StorageIndex>>
struct SparseLUVisitor : public boost::python::def_visitor<
                             SparseLUVisitor<_MatrixType, _Ordering>> {
  typedef SparseLUVisitor<_MatrixType, _Ordering> Visitor;
  typedef _MatrixType MatrixType;

  typedef Eigen::SparseLU<MatrixType> Solver;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;

  typedef typename Solver::SCMatrix SCMatrix;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef Eigen::MappedSparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>
      MappedSparseMatrix;
  typedef Eigen::SparseLUMatrixLReturnType<SCMatrix> LType;
  typedef Eigen::SparseLUMatrixUReturnType<SCMatrix, MappedSparseMatrix> UType;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(bp::init<>(bp::arg("self"), "Default constructor"))
        .def(bp::init<MatrixType>(bp::args("self", "matrix"),
                                  "Constructs and performs the LU "
                                  "factorization from a given matrix."))

        .def("determinant", &Solver::determinant, bp::arg("self"),
             "Returns the determinant of the matrix.")
        .def("signDeterminant", &Solver::signDeterminant, bp::arg("self"),
             "A number representing the sign of the determinant. ")
        .def("absDeterminant", &Solver::absDeterminant, bp::arg("self"),
             "Returns the absolute value of the determinant of the matrix of "
             "which *this is the QR decomposition.")
        .def("logAbsDeterminant", &Solver::logAbsDeterminant, bp::arg("self"),
             "Returns the natural log of the absolute value of the determinant "
             "of the matrix of which *this is the QR decomposition")

        .def("rows", &Solver::rows, bp::arg("self"),
             "Returns the number of rows of the matrix.")
        .def("cols", &Solver::cols, bp::arg("self"),
             "Returns the number of cols of the matrix.")

        .def("nnzL", &Solver::nnzL, bp::arg("self"),
             "The number of non zero elements in L")
        .def("nnzU", &Solver::nnzU, bp::arg("self"),
             "The number of non zero elements in U")

        .def(
            "matrixL",
            +[](const Solver &self) -> LType { return self.matrixL(); },
            "Returns an expression of the matrix L.")
        .def(
            "matrixU",
            +[](const Solver &self) -> UType { return self.matrixU(); },
            "Returns an expression of the matrix U.")

        .def("colsPermutation", &Solver::colsPermutation, bp::arg("self"),
             "Returns a reference to the column matrix permutation PTc such "
             "that Pr A PTc = LU.",
             bp::return_value_policy<bp::copy_const_reference>())
        .def("rowsPermutation", &Solver::rowsPermutation, bp::arg("self"),
             "Returns a reference to the row matrix permutation Pr such that "
             "Pr A PTc = LU",
             bp::return_value_policy<bp::copy_const_reference>())

        .def("compute", &Solver::compute, bp::args("self", "matrix"),
             "Compute the symbolic and numeric factorization of the input "
             "sparse matrix. "
             "The input matrix should be in column-major storage. ")

        .def("analyzePattern", &Solver::analyzePattern,
             bp::args("self", "matrix"),
             "Compute the column permutation to minimize the fill-in.")
        .def("factorize", &Solver::factorize, bp::args("self", "matrix"),
             "Performs a numeric decomposition of a given matrix.\n"
             "The given matrix must has the same sparcity than the matrix on "
             "which the symbolic decomposition has been performed.")

        .def("isSymmetric", &Solver::isSymmetric, bp::args("self", "sym"),
             "Indicate that the pattern of the input matrix is symmetric. ")

        .def("setPivotThreshold", &Solver::setPivotThreshold,
             bp::args("self", "thresh"),
             "Set the threshold used for a diagonal entry to be an acceptable "
             "pivot.")

        .def("info", &Solver::info, bp::arg("self"),
             "NumericalIssue if the input contains INF or NaN values or "
             "overflow occured. Returns Success otherwise.")
        .def("lastErrorMessage", &Solver::lastErrorMessage, bp::arg("self"),
             "Returns a string describing the type of error. ")

        .def(SparseSolverBaseVisitor<Solver>());
  }

  static void expose() {
    static const std::string classname =
        "SparseLU_" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string &name) {
    bp::class_<Solver, boost::noncopyable>(
        name.c_str(),
        "Sparse supernodal LU factorization for general matrices. \n\n"
        "This class implements the supernodal LU factorization for general "
        "matrices. "
        "It uses the main techniques from the sequential SuperLU package "
        "(http://crd-legacy.lbl.gov/~xiaoye/SuperLU/). It handles "
        "transparently real "
        "and complex arithmetic with single and double precision, depending on "
        "the scalar "
        "type of your input matrix. The code has been optimized to provide "
        "BLAS-3 operations "
        "during supernode-panel updates. It benefits directly from the "
        "built-in high-performant "
        "Eigen BLAS routines. Moreover, when the size of a supernode is very "
        "small, the BLAS "
        "calls are avoided to enable a better optimization from the compiler. "
        "For best performance, "
        "you should compile it with NDEBUG flag to avoid the numerous bounds "
        "checking on vectors.",
        bp::no_init)
        .def(SparseLUVisitor())
        .def(IdVisitor<Solver>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_sparse_lu_hpp__
