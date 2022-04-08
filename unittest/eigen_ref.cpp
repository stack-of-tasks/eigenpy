/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2022, INRIA
 */

#include <iostream>

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/eigen-from-python.hpp"

using namespace Eigen;
using namespace eigenpy;

template <typename MatType>
void printMatrix(const Eigen::Ref<const MatType> mat) {
  if (MatType::IsVectorAtCompileTime) std::cout << "isVector" << std::endl;
  std::cout << "size: cols " << mat.cols() << " rows " << mat.rows()
            << std::endl;
  std::cout << mat << std::endl;
}

template <typename VecType>
void printVector(const Eigen::Ref<const VecType>& vec) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecType);
  printMatrix(vec);
}

template <typename MatType>
void setOnes(Eigen::Ref<MatType> mat) {
  mat.setOnes();
}

template <typename MatType>
void fill(Eigen::Ref<MatType> mat, const typename MatType::Scalar& value) {
  mat.fill(value);
}

template <typename MatType>
Eigen::Ref<MatType> asRef(const int rows, const int cols) {
  static MatType mat(rows, cols);
  std::cout << "mat:\n" << mat << std::endl;
  return mat;
}

template <typename MatType>
Eigen::Ref<MatType> asRef(Eigen::Ref<MatType> mat) {
  return Eigen::Ref<MatType>(mat);
}

template <typename MatType>
const Eigen::Ref<const MatType> asConstRef(Eigen::Ref<MatType> mat) {
  return Eigen::Ref<const MatType>(mat);
}

BOOST_PYTHON_MODULE(eigen_ref) {
  namespace bp = boost::python;
  eigenpy::enableEigenPy();

  bp::def("printMatrix", printMatrix<Vector3d>);
  bp::def("printMatrix", printMatrix<VectorXd>);
  bp::def("printMatrix", printMatrix<MatrixXd>);

  bp::def("printVector", printVector<VectorXd>);
  bp::def("printRowVector", printVector<RowVectorXd>);

  bp::def("setOnes", setOnes<Vector3d>);
  bp::def("setOnes", setOnes<VectorXd>);
  bp::def("setOnes", setOnes<MatrixXd>);

  bp::def("fillVec3", fill<Vector3d>);
  bp::def("fillVec", fill<VectorXd>);
  bp::def("fill", fill<MatrixXd>);

  bp::def("asRef",
          (Eigen::Ref<MatrixXd>(*)(const int, const int))asRef<MatrixXd>);
  bp::def("asRef",
          (Eigen::Ref<MatrixXd>(*)(Eigen::Ref<MatrixXd>))asRef<MatrixXd>);
  bp::def("asConstRef", (const Eigen::Ref<const MatrixXd> (*)(
                            Eigen::Ref<MatrixXd>))asConstRef<MatrixXd>);
}
