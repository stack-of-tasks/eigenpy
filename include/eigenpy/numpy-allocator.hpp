/*
 * Copyright 2020 INRIA
 */

#ifndef __eigenpy_numpy_allocator_hpp__
#define __eigenpy_numpy_allocator_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/numpy-type.hpp"
#include "eigenpy/eigen-allocator.hpp"

#include "eigenpy/register.hpp"

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
      
      const int code = Register::getTypeCode<Scalar>();
      PyArrayObject * pyArray = (PyArrayObject*) call_PyArray_SimpleNew(static_cast<int>(nd), shape, code);
      
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
        const int Scalar_type_code = Register::getTypeCode<Scalar>();
        PyArrayObject * pyArray = (PyArrayObject*) call_PyArray_New(getPyArrayType(),
                                                                    static_cast<int>(nd),
                                                                    shape,
                                                                    Scalar_type_code,
                                                                    mat.data(),
                                                                    NPY_ARRAY_MEMORY_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        
        return pyArray;
      }
      else
      {
        return NumpyAllocator<MatType>::allocate(mat,nd,shape);
      }
    }
  };

#if EIGEN_VERSION_AT_LEAST(3,2,0)

  template<typename MatType, int Options, typename Stride>
  struct NumpyAllocator<Eigen::Ref<MatType,Options,Stride> >
  {
    typedef Eigen::Ref<MatType,Options,Stride> RefType;
    
    static PyArrayObject * allocate(RefType & mat,
                                    npy_intp nd, npy_intp * shape)
    {
      typedef typename RefType::Scalar Scalar;
      enum { NPY_ARRAY_MEMORY_CONTIGUOUS = RefType::IsRowMajor ? NPY_ARRAY_CARRAY : NPY_ARRAY_FARRAY };
      
      if(NumpyType::sharedMemory())
      {
        const int Scalar_type_code = Register::getTypeCode<Scalar>();
        PyArrayObject * pyArray = (PyArrayObject*) call_PyArray_New(getPyArrayType(),
                                                                    static_cast<int>(nd),
                                                                    shape,
                                                                    Scalar_type_code,
                                                                    mat.data(),
                                                                    NPY_ARRAY_MEMORY_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        
        return pyArray;
      }
      else
      {
        return NumpyAllocator<MatType>::allocate(mat,nd,shape);
      }
    }
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
        const int Scalar_type_code = Register::getTypeCode<Scalar>();
        PyArrayObject * pyArray = (PyArrayObject*) call_PyArray_New(getPyArrayType(),
                                                                    static_cast<int>(nd),
                                                                    shape,
                                                                    Scalar_type_code,
                                                                    const_cast<Scalar *>(mat.data()),
                                                                    NPY_ARRAY_MEMORY_CONTIGUOUS_RO | NPY_ARRAY_ALIGNED);
                                                                    
        return pyArray;
      }
      else
      {
        return NumpyAllocator<MatType>::allocate(mat,nd,shape);
      }
    }
  };

#if EIGEN_VERSION_AT_LEAST(3,2,0)

  template<typename MatType, int Options, typename Stride>
  struct NumpyAllocator<const Eigen::Ref<const MatType,Options,Stride> >
  {
    typedef const Eigen::Ref<const MatType,Options,Stride> RefType;
    
    template<typename SimilarMatrixType>
    static PyArrayObject * allocate(RefType & mat,
                                    npy_intp nd, npy_intp * shape)
    {
      typedef typename SimilarMatrixType::Scalar Scalar;
      enum { NPY_ARRAY_MEMORY_CONTIGUOUS_RO = SimilarMatrixType::IsRowMajor ? NPY_ARRAY_CARRAY_RO : NPY_ARRAY_FARRAY_RO };
      
      if(NumpyType::sharedMemory())
      {
        const int Scalar_type_code = Register::getTypeCode<Scalar>();
        PyArrayObject * pyArray = (PyArrayObject*) call_PyArray_New(getPyArrayType(),
                                                                    static_cast<int>(nd),
                                                                    shape,
                                                                    Scalar_type_code,
                                                                    const_cast<Scalar *>(mat.data()),
                                                                    NPY_ARRAY_MEMORY_CONTIGUOUS_RO | NPY_ARRAY_ALIGNED);
                                                                    
        return pyArray;
      }
      else
      {
        return NumpyAllocator<MatType>::allocate(mat,nd,shape);
      }
    }
  };

#endif
}

#endif // ifndef __eigenpy_numpy_allocator_hpp__
