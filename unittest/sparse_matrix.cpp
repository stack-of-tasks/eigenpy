/*
 * Copyright 2024 CNRS INRIA
 */

#include <iostream>

#include "eigenpy/eigenpy.hpp"

template <typename Scalar, int Options>
Eigen::SparseMatrix<Scalar, Options> vector1x1(const Scalar& value) {
  typedef Eigen::SparseMatrix<Scalar, Options> ReturnType;
  ReturnType mat(1, 1);
  mat.coeffRef(0, 0) = value;
  mat.makeCompressed();
  return mat;
}

template <typename Scalar, int Options>
Eigen::SparseMatrix<Scalar, Options> matrix1x1(const Scalar& value) {
  typedef Eigen::SparseMatrix<Scalar, Options> ReturnType;
  ReturnType mat(1, 1);
  mat.coeffRef(0, 0) = value;
  mat.makeCompressed();
  return mat;
}

template <typename Scalar, int Options>
Eigen::SparseMatrix<Scalar, Options> diagonal(
    const Eigen::Ref<Eigen::Matrix<Scalar, Eigen::Dynamic, 1> >& diag_values) {
  typedef Eigen::SparseMatrix<Scalar, Options> ReturnType;
  ReturnType mat(diag_values.size(), diag_values.size());
  for (Eigen::Index k = 0; k < diag_values.size(); ++k)
    mat.coeffRef(k, k) = diag_values[k];
  mat.makeCompressed();
  return mat;
}

template <typename Scalar, int Options>
void matrix1x1_input(const Eigen::Matrix<Scalar, 1, 1>& mat) {
  std::cout << mat << std::endl;
}

template <typename Scalar, int Options>
Eigen::SparseMatrix<Scalar, Options> emptyVector() {
  return Eigen::SparseMatrix<Scalar, Options>();
}

template <typename Scalar, int Options>
Eigen::SparseMatrix<Scalar, Options> emptyMatrix() {
  return Eigen::SparseMatrix<Scalar, Options>();
}

template <typename Scalar, int Options>
void print(const Eigen::SparseMatrix<Scalar, Options>& mat) {
  std::cout << mat << std::endl;
}

template <typename Scalar, int Options>
Eigen::SparseMatrix<Scalar, Options> copy(
    const Eigen::SparseMatrix<Scalar, Options>& mat) {
  return mat;
}

BOOST_PYTHON_MODULE(sparse_matrix) {
  using namespace Eigen;
  namespace bp = boost::python;
  eigenpy::enableEigenPy();

  typedef Eigen::SparseMatrix<double> SparseMatrixD;
  eigenpy::EigenToPyConverter<SparseMatrixD>::registration();
  eigenpy::EigenFromPyConverter<SparseMatrixD>::registration();

  bp::def("vector1x1", vector1x1<double, Eigen::ColMajor>);
  bp::def("matrix1x1", matrix1x1<double, Eigen::ColMajor>);
  bp::def("matrix1x1", matrix1x1_input<double, Eigen::ColMajor>);
  bp::def("matrix1x1_int", matrix1x1_input<int, Eigen::ColMajor>);

  bp::def("print", print<double, Eigen::ColMajor>);
  bp::def("copy", copy<double, Eigen::ColMajor>);
  bp::def("diagonal", diagonal<double, Eigen::ColMajor>);

  bp::def("emptyVector", emptyVector<double, Eigen::ColMajor>);
  bp::def("emptyMatrix", emptyMatrix<double, Eigen::ColMajor>);
}
