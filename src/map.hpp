/*
 * Copyright 2014, Nicolas Mansard, LAAS-CNRS
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

#include <Eigen/Core>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <eigenpy/exception.hpp>

namespace eigenpy
{
  template< typename MatType, int IsVector>
  struct MapNumpyTraits {};
 
  /* Wrap a numpy::array with an Eigen::Map. No memory copy. */
  template< typename MatType >
  struct MapNumpy
  {
    typedef MapNumpyTraits<MatType, MatType::IsVectorAtCompileTime> Impl;
    typedef typename Impl::EigenMap EigenMap;

    static inline EigenMap map( PyArrayObject* pyArray );
   };

} // namespace eigenpy

/* --- DETAILS ------------------------------------------------------------------ */
/* --- DETAILS ------------------------------------------------------------------ */
/* --- DETAILS ------------------------------------------------------------------ */

namespace eigenpy
{
  template<typename MatType>
  struct MapNumpyTraits<MatType,0>
  {
    typedef Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic> Stride;
    typedef Eigen::Map<MatType,0,Stride> EigenMap;
    typedef typename MatType::Scalar T;

    static EigenMap mapImpl( PyArrayObject* pyArray )
    {
      assert( PyArray_NDIM(pyArray) == 2 );
      
      const int R = PyArray_DIMS(pyArray)[0];
      const int C = PyArray_DIMS(pyArray)[1];
      const int itemsize = PyArray_ITEMSIZE(pyArray);
      const int stride1 = PyArray_STRIDE(pyArray, 0) / itemsize;
      const int stride2 = PyArray_STRIDE(pyArray, 1) / itemsize;
      
      if( (MatType::RowsAtCompileTime!=R)
	  && (MatType::RowsAtCompileTime!=Eigen::Dynamic) )
	{ throw eigenpy::Exception("The number of rows does not fit with the matrix type."); }
      if( (MatType::ColsAtCompileTime!=C)
	  && (MatType::ColsAtCompileTime!=Eigen::Dynamic) )
	{  throw eigenpy::Exception("The number of columns does not fit with the matrix type."); }

      T* pyData = reinterpret_cast<T*>(PyArray_DATA(pyArray));
      return EigenMap( pyData, R,C, Stride(stride2,stride1) );
    }
  };

  template<typename MatType>
  struct MapNumpyTraits<MatType,1>
  {
    typedef Eigen::InnerStride<Eigen::Dynamic> Stride;
    typedef Eigen::Map<MatType,0,Stride> EigenMap;
    typedef typename MatType::Scalar T;
 
    static EigenMap mapImpl( PyArrayObject* pyArray )
    {
      assert( PyArray_NDIM(pyArray) <= 2 );

      int rowMajor;
      if(  PyArray_NDIM(pyArray)==1 ) rowMajor = 0;
      else rowMajor = (PyArray_DIMS(pyArray)[0]>PyArray_DIMS(pyArray)[1])?0:1;

      const int R = PyArray_DIMS(pyArray)[rowMajor];
      const int itemsize = PyArray_ITEMSIZE(pyArray);
      const int stride = PyArray_STRIDE(pyArray, rowMajor) / itemsize;;

      if( (MatType::MaxSizeAtCompileTime!=R)
	      && (MatType::MaxSizeAtCompileTime!=Eigen::Dynamic) )
	{ throw eigenpy::Exception("The number of elements does not fit with the vector type."); }

      T* pyData = reinterpret_cast<T*>(PyArray_DATA(pyArray));
      return EigenMap( pyData, R, 1, Stride(stride) );
    }
  };

  template< typename MatType >
  typename MapNumpy<MatType>::EigenMap MapNumpy<MatType>::map( PyArrayObject* pyArray )
  {
    return Impl::mapImpl(pyArray); 
  }

} // namespace eigenpy
