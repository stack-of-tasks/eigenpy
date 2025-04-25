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
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>&
        diag_values) {
  typedef Eigen::SparseMatrix<Scalar, Options> ReturnType;
  ReturnType mat(diag_values.size(), diag_values.size());
  for (Eigen::Index k = 0; k < diag_values.size(); ++k)
    mat.coeffRef(k, k) = diag_values[k];
  mat.makeCompressed();
  return mat;
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

template <typename Scalar, int Options>
void expose_functions() {
  namespace bp = boost::python;
  bp::def("vector1x1", vector1x1<Scalar, Options>);
  bp::def("matrix1x1", matrix1x1<Scalar, Options>);

  bp::def("print", print<Scalar, Options>);
  bp::def("copy", copy<Scalar, Options>);
  bp::def("diagonal", diagonal<Scalar, Options>);

  bp::def("emptyVector", emptyVector<Scalar, Options>);
  bp::def("emptyMatrix", emptyMatrix<Scalar, Options>);
}

BOOST_PYTHON_MODULE(sparse_matrix) {
  namespace bp = boost::python;
  eigenpy::enableEigenPy();

  expose_functions<double, Eigen::ColMajor>();
  expose_functions<double, Eigen::RowMajor>();
}
