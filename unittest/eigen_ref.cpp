/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2020, INRIA
 */

#include "eigenpy/eigenpy.hpp"
#include <iostream>

using namespace Eigen;
using namespace eigenpy;

template<typename MatType>
void printMatrix(const Eigen::Ref<MatType> & mat)
{
  if(MatType::IsVectorAtCompileTime)
    std::cout << "isVector" << std::endl;
  std::cout << "size: cols " << mat.cols() << " rows " << mat.rows() << std::endl;
  std::cout << mat << std::endl;
}

template<typename VecType>
void printVector(const Eigen::Ref<VecType> & vec)
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecType);
  printMatrix(vec);
}

template<typename MatType>
void setOnes(Eigen::Ref<MatType> mat)
{
  printMatrix(mat);
  mat.setOnes();
  printMatrix(mat);
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
}
