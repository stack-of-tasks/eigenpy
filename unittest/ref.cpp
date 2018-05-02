/*
 * Copyright 2018, Justin Carpentier <jcarpent@laas.fr>, LAAS-CNRS
 *
 * This file is part of eigenpy.
 * eigenpy is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 * eigenpy is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.  You should
 * have received a copy of the GNU Lesser General Public License along
 * with eigenpy.  If not, see <http://www.gnu.org/licenses/>.
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



BOOST_PYTHON_MODULE(ref)
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
