/*
 * Copyright 2020 INRIA
 */

#ifndef __eigenpy_numpy_allocator_hpp__
#define __eigenpy_numpy_allocator_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/numpy-type.hpp"
#include "eigenpy/eigen-allocator.hpp"

namespace eigenpy
{
  template<typename MatType>
  struct NumpyAllocator
  {
    template<typename SimilarMatrixType>
    static PyArrayObject * allocate(const Eigen::MatrixBase<SimilarMatrixType> & mat,
                                    npy_intp nd, npy_intp * shape)
    {
      typedef typename SimilarMatrixType::Scalar Scalar;
      
      PyArrayObject * pyArray = (PyArrayObject*) call_PyArray_SimpleNew(static_cast<int>(nd), shape,
                                                                        NumpyEquivalentType<Scalar>::type_code);
      
      // Copy data
      EigenAllocator<SimilarMatrixType>::copy(mat,pyArray);
      
      return pyArray;
    }
  };

  template<typename MatType>
  struct NumpyAllocator<MatType &>
  {
    template<typename SimilarMatrixType>
    static PyArrayObject * allocate(Eigen::PlainObjectBase<SimilarMatrixType> & mat,
                                    npy_intp nd, npy_intp * shape)
    {
      typedef typename SimilarMatrixType::Scalar Scalar;
      enum { NPY_ARRAY_MEMORY_CONTIGUOUS = SimilarMatrixType::IsRowMajor ? NPY_ARRAY_CARRAY : NPY_ARRAY_FARRAY };
      
      if(NumpyType::sharedMemory())
      {
        PyTypeObject * py_type_ptr = &PyArray_Type;
        PyArrayObject * pyArray = (PyArrayObject*) call_PyArray_New(py_type_ptr,
                                                                    static_cast<int>(nd),
                                                                    shape,
                                                                    NumpyEquivalentType<Scalar>::type_code,
                                                                    mat.data(),
                                                                    NPY_ARRAY_MEMORY_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        
        return pyArray;
      }
      else
      {
        return NumpyAllocator<MatType>::allocate(mat.derived(),nd,shape);
      }
    }
  };

#if EIGEN_VERSION_AT_LEAST(3,2,0)

  template<typename MatType, int Options, typename Stride>
  struct NumpyAllocator<Eigen::Ref<MatType,Options,Stride> > : NumpyAllocator<MatType &>
  {
  };

#endif

  template<typename MatType>
  struct NumpyAllocator<const MatType &>
  {
    template<typename SimilarMatrixType>
    static PyArrayObject * allocate(const Eigen::PlainObjectBase<SimilarMatrixType> & mat,
                                    npy_intp nd, npy_intp * shape)
    {
      typedef typename SimilarMatrixType::Scalar Scalar;
      enum { NPY_ARRAY_MEMORY_CONTIGUOUS_RO = SimilarMatrixType::IsRowMajor ? NPY_ARRAY_CARRAY_RO : NPY_ARRAY_FARRAY_RO };
      
      if(NumpyType::sharedMemory())
      {
        PyTypeObject * py_type_ptr = &PyArray_Type;
        PyArrayObject * pyArray = (PyArrayObject*) call_PyArray_New(py_type_ptr,
                                                                    static_cast<int>(nd),
                                                                    shape,
                                                                    NumpyEquivalentType<Scalar>::type_code,
                                                                    const_cast<SimilarMatrixType &>(mat.derived()).data(),
                                                                    NPY_ARRAY_MEMORY_CONTIGUOUS_RO | NPY_ARRAY_ALIGNED);
                                                                    
        return pyArray;
      }
      else
      {
        return NumpyAllocator<MatType>::allocate(mat.derived(),nd,shape);
      }
    }
  };

#if EIGEN_VERSION_AT_LEAST(3,2,0)

  template<typename MatType, int Options, typename Stride>
  struct NumpyAllocator<const Eigen::Ref<const MatType,Options,Stride> > : NumpyAllocator<const MatType &>
  {
  };

#endif
}

#endif // ifndef __eigenpy_numpy_allocator_hpp__
