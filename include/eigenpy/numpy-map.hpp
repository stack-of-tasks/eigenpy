/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2020, INRIA
 */

#ifndef __eigenpy_numpy_map_hpp__
#define __eigenpy_numpy_map_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/exception.hpp"
#include "eigenpy/stride.hpp"

namespace eigenpy
{
  template<typename MatType, typename InputScalar, int AlignmentValue, typename Stride, bool IsVector = MatType::IsVectorAtCompileTime>
  struct NumpyMapTraits {};
 
  /* Wrap a numpy::array with an Eigen::Map. No memory copy. */
  template<typename MatType, typename InputScalar, int AlignmentValue = EIGENPY_NO_ALIGNMENT_VALUE, typename Stride = typename StrideType<MatType>::type>
  struct NumpyMap
  {
    typedef NumpyMapTraits<MatType, InputScalar, AlignmentValue, Stride> Impl;
    typedef typename Impl::EigenMap EigenMap;

    static EigenMap map(PyArrayObject* pyArray, bool swap_dimensions = false);
   };

} // namespace eigenpy

/* --- DETAILS ------------------------------------------------------------------ */
/* --- DETAILS ------------------------------------------------------------------ */
/* --- DETAILS ------------------------------------------------------------------ */

namespace eigenpy
{
  template<typename MatType, typename InputScalar, int AlignmentValue, typename Stride>
  struct NumpyMapTraits<MatType,InputScalar,AlignmentValue,Stride,false>
  {
    typedef Eigen::Matrix<InputScalar,MatType::RowsAtCompileTime,MatType::ColsAtCompileTime,MatType::Options> EquivalentInputMatrixType;
    typedef Eigen::Map<EquivalentInputMatrixType,AlignmentValue,Stride> EigenMap;

    static EigenMap mapImpl(PyArrayObject* pyArray, bool swap_dimensions = false)
    {
      enum {
        OuterStrideAtCompileTime = Stride::OuterStrideAtCompileTime,
        InnerStrideAtCompileTime = Stride::InnerStrideAtCompileTime,
      };
      
      assert(PyArray_NDIM(pyArray) == 2 ||  PyArray_NDIM(pyArray) == 1);
    
      const long int itemsize = PyArray_ITEMSIZE(pyArray);
      int inner_stride = -1, outer_stride = -1;
      int rows = -1, cols = -1;
      if(PyArray_NDIM(pyArray) == 2)
      {
        assert( (PyArray_DIMS(pyArray)[0]  < INT_MAX)
             && (PyArray_DIMS(pyArray)[1]  < INT_MAX)
             && (PyArray_STRIDE(pyArray,0) < INT_MAX)
             && (PyArray_STRIDE(pyArray,1) < INT_MAX) );
        
        rows = (int)PyArray_DIMS(pyArray)[0];
        cols = (int)PyArray_DIMS(pyArray)[1];
        
        if(EquivalentInputMatrixType::IsRowMajor)
        {
          inner_stride = (int)PyArray_STRIDE(pyArray, 1) / (int)itemsize;
          outer_stride = (int)PyArray_STRIDE(pyArray, 0) / (int)itemsize;
        }
        else
        {
          inner_stride = (int)PyArray_STRIDE(pyArray, 0) / (int)itemsize;
          outer_stride = (int)PyArray_STRIDE(pyArray, 1) / (int)itemsize;
        }
      }
      else if(PyArray_NDIM(pyArray) == 1)
      {
        assert( (PyArray_DIMS(pyArray)[0]  < INT_MAX)
             && (PyArray_STRIDE(pyArray,0) < INT_MAX));
        
        if(!swap_dimensions)
        {
          rows = (int)PyArray_DIMS(pyArray)[0];
          cols = 1;
          
          if(EquivalentInputMatrixType::IsRowMajor)
          {
            outer_stride = (int)PyArray_STRIDE(pyArray, 0) / (int)itemsize;
            inner_stride = 0;
          }
          else
          {
            inner_stride = (int)PyArray_STRIDE(pyArray, 0) / (int)itemsize;
            outer_stride = 0;
          }
        }
        else
        {
          rows = 1;
          cols = (int)PyArray_DIMS(pyArray)[0];
          
          if(EquivalentInputMatrixType::IsRowMajor)
          {
            inner_stride = (int)PyArray_STRIDE(pyArray, 0) / (int)itemsize;
            outer_stride = 0;
          }
          else
          {
            inner_stride = 0;
            outer_stride = (int)PyArray_STRIDE(pyArray, 0) / (int)itemsize;
          }
        }
      }
      
      // Specific care for Eigen::Stride<-1,0>
      if(InnerStrideAtCompileTime==0 && OuterStrideAtCompileTime==Eigen::Dynamic)
      {
        outer_stride = std::max(inner_stride,outer_stride); inner_stride = 0;
      }
      
      Stride stride(OuterStrideAtCompileTime==Eigen::Dynamic?outer_stride:OuterStrideAtCompileTime,
                    InnerStrideAtCompileTime==Eigen::Dynamic?inner_stride:InnerStrideAtCompileTime);

      if(   (MatType::RowsAtCompileTime != rows)
         && (MatType::RowsAtCompileTime != Eigen::Dynamic) )
      { throw eigenpy::Exception("The number of rows does not fit with the matrix type."); }
      
      if(   (MatType::ColsAtCompileTime != cols)
         && (MatType::ColsAtCompileTime != Eigen::Dynamic) )
      {  throw eigenpy::Exception("The number of columns does not fit with the matrix type."); }
      
      InputScalar* pyData = reinterpret_cast<InputScalar*>(PyArray_DATA(pyArray));
      
      return EigenMap(pyData, rows, cols, stride);
    }
  };

  template<typename MatType, typename InputScalar, int AlignmentValue, typename Stride>
  struct NumpyMapTraits<MatType,InputScalar,AlignmentValue,Stride,true>
  {
    typedef Eigen::Matrix<InputScalar,MatType::RowsAtCompileTime,MatType::ColsAtCompileTime,MatType::Options> EquivalentInputMatrixType;
    typedef Eigen::Map<EquivalentInputMatrixType,AlignmentValue,Stride> EigenMap;
 
    static EigenMap mapImpl(PyArrayObject* pyArray, bool swap_dimensions = false)
    {
      EIGENPY_UNUSED_VARIABLE(swap_dimensions);
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

      if(   (MatType::MaxSizeAtCompileTime != R)
         && (MatType::MaxSizeAtCompileTime != Eigen::Dynamic) )
      { throw eigenpy::Exception("The number of elements does not fit with the vector type."); }

      InputScalar* pyData = reinterpret_cast<InputScalar*>(PyArray_DATA(pyArray));
      
      return EigenMap( pyData, R, Stride(stride) );
    }
  };

  template<typename MatType, typename InputScalar, int AlignmentValue, typename Stride>
  typename NumpyMap<MatType,InputScalar,AlignmentValue,Stride>::EigenMap
  NumpyMap<MatType,InputScalar,AlignmentValue,Stride>::map(PyArrayObject * pyArray, bool swap_dimensions)
  {
    return Impl::mapImpl(pyArray,swap_dimensions);
  }

} // namespace eigenpy

#endif // define __eigenpy_numpy_map_hpp__
