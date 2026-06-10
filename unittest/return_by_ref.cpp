/*
 * Copyright 2020 INRIA
 */

#include <iostream>

#include "eigenpy/eigenpy.hpp"

template <typename Matrix>
struct Base {
  Base(const Eigen::DenseIndex rows, const Eigen::DenseIndex cols)
      : mat(rows, cols) {}

  void show() { std::cout << mat << std::endl; }

  Matrix& ref() { return mat; }
  const Matrix& const_ref() { return mat; }
  Matrix copy() { return mat; }

 protected:
  Matrix mat;
};

template <typename MatrixType>
void expose_matrix_class(const std::string& name) {
  using namespace Eigen;
  namespace bp = boost::python;

  bp::class_<Base<MatrixType>>(name.c_str(), bp::init<DenseIndex, DenseIndex>())
      .def("show", &Base<MatrixType>::show)
      .def("ref", &Base<MatrixType>::ref, bp::return_internal_reference<>())
      .def("const_ref", &Base<MatrixType>::const_ref,
           bp::return_internal_reference<>())
      .def("copy", &Base<MatrixType>::copy);
}

BOOST_PYTHON_MODULE(return_by_ref) {
  using namespace Eigen;
  eigenpy::enableEigenPy();

  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorType;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      RowMatrixType;

  expose_matrix_class<VectorType>("Vector");
  expose_matrix_class<MatrixType>("Matrix");
  expose_matrix_class<RowMatrixType>("RowMatrix");
}
