/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2020, INRIA
 */

#include "eigenpy/eigenpy.hpp"
#include <iostream>

template<typename Matrix>
struct Base
{
  Base(const Eigen::DenseIndex rows,
       const Eigen::DenseIndex cols)
  : mat(rows,cols)
  {}
  
  void show()
  {
    std::cout << mat << std::endl;
  }
  
  Matrix & ref() { return mat; }
  Matrix  copy() { return mat; }
  
protected:
  
  Matrix mat;
};


BOOST_PYTHON_MODULE(return_by_ref)
{
  using namespace Eigen;
  namespace bp = boost::python;
  eigenpy::enableEigenPy();

  
  typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixType;
  bp::class_<Base<RowMatrixType> >("Matrix",bp::init<DenseIndex,DenseIndex>())
  .def("show",&Base<RowMatrixType>::show)
  .def("ref",&Base<RowMatrixType>::ref, bp::return_internal_reference<>())
  .def("copy",&Base<RowMatrixType>::copy);
}

