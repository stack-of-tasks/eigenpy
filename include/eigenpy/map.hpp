/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2020, INRIA
 */

#include "eigenpy/fwd.hpp"
#include <numpy/arrayobject.h>
#include "eigenpy/exception.hpp"
#include "eigenpy/stride.hpp"

namespace eigenpy
{
  template<typename MatType, typename InputScalar, bool IsVector>
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
  struct MapNumpyTraits<MatType,InputScalar,false>
  {
    typedef typename StrideType<MatType>::type Stride;
    typedef Eigen::Matrix<InputScalar,MatType::RowsAtCompileTime,MatType::ColsAtCompileTime> EquivalentInputMatrixType;
    typedef Eigen::Map<EquivalentInputMatrixType,EIGENPY_DEFAULT_ALIGNMENT_VALUE,Stride> EigenMap;

    static EigenMap mapImpl( PyArrayObject* pyArray )
    {
      assert(PyArray_NDIM(pyArray) == 2 ||  PyArray_NDIM(pyArray) == 1);
    
      const long int itemsize = PyArray_ITEMSIZE(pyArray);
      int stride1 = -1, stride2 = -1;
      int rows = -1, cols = -1;
      if(PyArray_NDIM(pyArray) == 2)
      {
        assert( (PyArray_DIMS(pyArray)[0]  < INT_MAX)
             && (PyArray_DIMS(pyArray)[1]  < INT_MAX)
             && (PyArray_STRIDE(pyArray,0) < INT_MAX)
             && (PyArray_STRIDE(pyArray,1) < INT_MAX) );
        
        rows = (int)PyArray_DIMS(pyArray)[0];
        cols = (int)PyArray_DIMS(pyArray)[1];
        stride1 = (int)PyArray_STRIDE(pyArray, 0) / (int)itemsize;
        stride2 = (int)PyArray_STRIDE(pyArray, 1) / (int)itemsize;
      }
      else if(PyArray_NDIM(pyArray) == 1)
      {
        assert( (PyArray_DIMS(pyArray)[0]  < INT_MAX)
             && (PyArray_STRIDE(pyArray,0) < INT_MAX));
        
        rows = (int)PyArray_DIMS(pyArray)[0];
        cols = 1;
        
        stride1 = (int)PyArray_STRIDE(pyArray, 0) / (int)itemsize;
        stride2 = 0;
      }
      
      Stride stride(stride2,stride1);
      
      if( (MatType::RowsAtCompileTime!=rows)
         && (MatType::RowsAtCompileTime!=Eigen::Dynamic) )
      { throw eigenpy::Exception("The number of rows does not fit with the matrix type."); }
      if( (MatType::ColsAtCompileTime!=cols)
         && (MatType::ColsAtCompileTime!=Eigen::Dynamic) )
      {  throw eigenpy::Exception("The number of columns does not fit with the matrix type."); }
      
      InputScalar* pyData = reinterpret_cast<InputScalar*>(PyArray_DATA(pyArray));
      
      return EigenMap( pyData, rows, cols, stride );
    }
  };

  template<typename MatType, typename InputScalar>
  struct MapNumpyTraits<MatType,InputScalar,true>
  {
    typedef typename StrideType<MatType>::type Stride;
    typedef Eigen::Matrix<InputScalar,MatType::RowsAtCompileTime,MatType::ColsAtCompileTime> EquivalentInputMatrixType;
    typedef Eigen::Map<EquivalentInputMatrixType,EIGENPY_DEFAULT_ALIGNMENT_VALUE,Stride> EigenMap;
 
    static EigenMap mapImpl( PyArrayObject* pyArray )
    {
      assert( PyArray_NDIM(pyArray) <= 2 );

      int rowMajor;
      if(  PyArray_NDIM(pyArray)==1 ) rowMajor = 0;
      else if (PyArray_DIMS(pyArray)[0] == 0) rowMajor = 0; // handle zero-size vector
      else if (PyArray_DIMS(pyArray)[1] == 0) rowMajor = 1; // handle zero-size vector
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
  typename MapNumpy<MatType,InputScalar>::EigenMap
  MapNumpy<MatType,InputScalar>::map(PyArrayObject * pyArray)
  {
    return Impl::mapImpl(pyArray); 
  }

} // namespace eigenpy
