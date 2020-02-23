/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2020, INRIA
 */

#include "eigenpy/eigenpy.hpp"
#include <iostream>

using namespace Eigen;
using namespace eigenpy;

template<typename MatType>
void printMatrix(const eigenpy::Ref<MatType> & mat)
{
  if(MatType::IsVectorAtCompileTime)
    std::cout << "isVector" << std::endl;
  std::cout << "size: cols " << mat.cols() << " rows " << mat.rows() << std::endl;
  std::cout << mat << std::endl;
}

template<typename MatType>
void printVector(const eigenpy::Ref<MatType> & mat)
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(MatType);
  printMatrix(mat);
}

template<typename MatType,int Options, typename StrideType>
void setOnes(Eigen::Ref<MatType,Options,StrideType> mat)
{
  mat.setOnes();
}

template<typename MatType>
void setOnes_wrap(eigenpy::Ref<MatType> mat)
{
  setOnes(mat);
}

BOOST_PYTHON_MODULE(eigenpy_ref)
{
  namespace bp = boost::python;
  eigenpy::enableEigenPy();
  
  bp::def("printMatrix", printMatrix<Vector3d>);
  bp::def("printMatrix", printMatrix<VectorXd>);
  bp::def("printMatrix", printMatrix<MatrixXd>);
  
  bp::def("printVector", printVector<VectorXd>);

  bp::def("setOnes", setOnes_wrap<Vector3d>);
  bp::def("setOnes", setOnes_wrap<VectorXd>);
  bp::def("setOnes", setOnes_wrap<MatrixXd>);
}
