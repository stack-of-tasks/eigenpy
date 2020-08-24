/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2020, INRIA
 */

#include "eigenpy/eigenpy.hpp"
#include <iostream>

using namespace Eigen;
using namespace eigenpy;

template<typename MatType>
void printMatrix(const Eigen::Ref<const MatType> mat)
{
  if(MatType::IsVectorAtCompileTime)
    std::cout << "isVector" << std::endl;
  std::cout << "size: cols " << mat.cols() << " rows " << mat.rows() << std::endl;
  std::cout << mat << std::endl;
}

template<typename VecType>
void printVector(const Eigen::Ref<const VecType> & vec)
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecType);
  printMatrix(vec);
}

template<typename MatType>
void setOnes(Eigen::Ref<MatType> mat)
{
  mat.setOnes();
}

template<typename MatType>
void fill(Eigen::Ref<MatType> mat, const typename MatType::Scalar & value)
{
  mat.fill(value);
}

template<typename MatType>
Eigen::Ref<MatType> asRef(const int rows, const int cols)
{
  static MatType mat(rows,cols);
  std::cout << "mat:\n" << mat << std::endl;
  return mat;
}

BOOST_PYTHON_MODULE(eigen_ref)
{
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
  
  bp::def("asRef", asRef<MatrixXd>);
}
