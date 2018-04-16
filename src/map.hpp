/*
 * Copyright 2014-2018, Nicolas Mansard and Justin Carpentier, LAAS-CNRS
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

#include "eigenpy/fwd.hpp"
#include <numpy/arrayobject.h>
#include "eigenpy/exception.hpp"
#include "eigenpy/stride.hpp"

namespace eigenpy
{
  template<typename MatType, typename InputScalar, int IsVector>
  struct MapNumpyTraits {};
 
  /* Wrap a numpy::array with an Eigen::Map. No memory copy. */
  template<typename MatType, typename InputScalar>
  struct MapNumpy
  {
    typedef MapNumpyTraits<MatType, InputScalar, MatType::IsVectorAtCompileTime> Impl;
    typedef typename Impl::EigenMap EigenMap;
    typedef typename Impl::Stride Stride;

    static inline EigenMap map( PyArrayObject* pyArray );
   };

} // namespace eigenpy

/* --- DETAILS ------------------------------------------------------------------ */
/* --- DETAILS ------------------------------------------------------------------ */
/* --- DETAILS ------------------------------------------------------------------ */

namespace eigenpy
{
  template<typename MatType, typename InputScalar>
  struct MapNumpyTraits<MatType,InputScalar,0>
  {
    typedef typename StrideType<MatType>::type Stride;
    typedef Eigen::Matrix<InputScalar,MatType::RowsAtCompileTime,MatType::ColsAtCompileTime> EquivalentInputMatrixType;
    typedef Eigen::Map<EquivalentInputMatrixType,EIGENPY_DEFAULT_ALIGNMENT_VALUE,Stride> EigenMap;

    static EigenMap mapImpl( PyArrayObject* pyArray )
    {
      assert( PyArray_NDIM(pyArray) == 2 );
      
      assert( (PyArray_DIMS(pyArray)[0]<INT_MAX)
	      && (PyArray_DIMS(pyArray)[1]<INT_MAX)
	      && (PyArray_STRIDE(pyArray, 0)<INT_MAX)
	      && (PyArray_STRIDE(pyArray, 1)<INT_MAX) );

      const int R = (int)PyArray_DIMS(pyArray)[0];
      const int C = (int)PyArray_DIMS(pyArray)[1];
      const long int itemsize = PyArray_ITEMSIZE(pyArray);
      const int stride1 = (int)PyArray_STRIDE(pyArray, 0) / (int)itemsize;
      const int stride2 = (int)PyArray_STRIDE(pyArray, 1) / (int)itemsize;
      Stride stride(stride2,stride1);
      
      
      
      if( (MatType::RowsAtCompileTime!=R)
         && (MatType::RowsAtCompileTime!=Eigen::Dynamic) )
      { throw eigenpy::Exception("The number of rows does not fit with the matrix type."); }
      if( (MatType::ColsAtCompileTime!=C)
         && (MatType::ColsAtCompileTime!=Eigen::Dynamic) )
      {  throw eigenpy::Exception("The number of columns does not fit with the matrix type."); }
      
      InputScalar* pyData = reinterpret_cast<InputScalar*>(PyArray_DATA(pyArray));
      
      return EigenMap( pyData, R,C, stride );
    }
  };

  template<typename MatType, typename InputScalar>
  struct MapNumpyTraits<MatType,InputScalar,1>
  {
    typedef typename StrideType<MatType>::type Stride;
    typedef Eigen::Matrix<InputScalar,MatType::RowsAtCompileTime,MatType::ColsAtCompileTime> EquivalentInputMatrixType;
    typedef Eigen::Map<EquivalentInputMatrixType,EIGENPY_DEFAULT_ALIGNMENT_VALUE,Stride> EigenMap;
 
    static EigenMap mapImpl( PyArrayObject* pyArray )
    {
      assert( PyArray_NDIM(pyArray) <= 2 );

      int rowMajor;
      if(  PyArray_NDIM(pyArray)==1 ) rowMajor = 0;
      else rowMajor = (PyArray_DIMS(pyArray)[0]>PyArray_DIMS(pyArray)[1])?0:1;

      assert( (PyArray_DIMS(pyArray)[rowMajor]< INT_MAX)
             && (PyArray_STRIDE(pyArray, rowMajor) ));
      const int R = (int)PyArray_DIMS(pyArray)[rowMajor];
      const long int itemsize = PyArray_ITEMSIZE(pyArray);
      const int stride = (int) PyArray_STRIDE(pyArray, rowMajor) / (int) itemsize;;

      if( (MatType::MaxSizeAtCompileTime!=R)
         && (MatType::MaxSizeAtCompileTime!=Eigen::Dynamic) )
      { throw eigenpy::Exception("The number of elements does not fit with the vector type."); }

      InputScalar* pyData = reinterpret_cast<InputScalar*>(PyArray_DATA(pyArray));
      
      return EigenMap( pyData, R, Stride(stride) );
    }
  };

  template<typename MatType, typename InputScalar>
  typename MapNumpy<MatType,InputScalar>::EigenMap MapNumpy<MatType,InputScalar>::map( PyArrayObject* pyArray )
  {
    return Impl::mapImpl(pyArray); 
  }

} // namespace eigenpy
