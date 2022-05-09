/*
 * Copyright 2020-2022 INRIA
 */

#ifndef __eigenpy_numpy_allocator_hpp__
#define __eigenpy_numpy_allocator_hpp__

#include "eigenpy/eigen-allocator.hpp"
#include "eigenpy/fwd.hpp"
#include "eigenpy/numpy-type.hpp"
#include "eigenpy/register.hpp"

namespace eigenpy {

template <typename MatType>
struct NumpyAllocator {
  template <typename SimilarMatrixType>
  static PyArrayObject *allocate(
      const Eigen::MatrixBase<SimilarMatrixType> &mat, npy_intp nd,
      npy_intp *shape) {
    typedef typename SimilarMatrixType::Scalar Scalar;

    const int code = Register::getTypeCode<Scalar>();
    PyArrayObject *pyArray = (PyArrayObject *)call_PyArray_SimpleNew(
        static_cast<int>(nd), shape, code);

    // Copy data
    EigenAllocator<SimilarMatrixType>::copy(mat, pyArray);

    return pyArray;
  }
};

template <typename MatType>
struct NumpyAllocator<MatType &> {
  template <typename SimilarMatrixType>
  static PyArrayObject *allocate(Eigen::PlainObjectBase<SimilarMatrixType> &mat,
                                 npy_intp nd, npy_intp *shape) {
    typedef typename SimilarMatrixType::Scalar Scalar;
    enum {
      NPY_ARRAY_MEMORY_CONTIGUOUS =
          SimilarMatrixType::IsRowMajor ? NPY_ARRAY_CARRAY : NPY_ARRAY_FARRAY
    };

    if (NumpyType::sharedMemory()) {
      const int Scalar_type_code = Register::getTypeCode<Scalar>();
      PyArrayObject *pyArray = (PyArrayObject *)call_PyArray_New(
          getPyArrayType(), static_cast<int>(nd), shape, Scalar_type_code,
          mat.data(), NPY_ARRAY_MEMORY_CONTIGUOUS | NPY_ARRAY_ALIGNED);

      return pyArray;
    } else {
      return NumpyAllocator<MatType>::allocate(mat, nd, shape);
    }
  }
};

#if EIGEN_VERSION_AT_LEAST(3, 2, 0)

template <typename MatType, int Options, typename Stride>
struct NumpyAllocator<Eigen::Ref<MatType, Options, Stride> > {
  typedef Eigen::Ref<MatType, Options, Stride> RefType;

  static PyArrayObject *allocate(RefType &mat, npy_intp nd, npy_intp *shape) {
    typedef typename RefType::Scalar Scalar;
    enum {
      NPY_ARRAY_MEMORY_CONTIGUOUS =
          RefType::IsRowMajor ? NPY_ARRAY_CARRAY : NPY_ARRAY_FARRAY
    };

    if (NumpyType::sharedMemory()) {
      const int Scalar_type_code = Register::getTypeCode<Scalar>();
      const bool reverse_strides = MatType::IsRowMajor || (mat.rows() == 1);
      Eigen::DenseIndex inner_stride = reverse_strides ? mat.outerStride()
                                                       : mat.innerStride(),
                        outer_stride = reverse_strides ? mat.innerStride()
                                                       : mat.outerStride();

      const int elsize = call_PyArray_DescrFromType(Scalar_type_code)->elsize;
      npy_intp strides[2] = {elsize * inner_stride, elsize * outer_stride};

      PyArrayObject *pyArray = (PyArrayObject *)call_PyArray_New(
          getPyArrayType(), static_cast<int>(nd), shape, Scalar_type_code,
          strides, mat.data(), NPY_ARRAY_MEMORY_CONTIGUOUS | NPY_ARRAY_ALIGNED);

      return pyArray;
    } else {
      return NumpyAllocator<MatType>::allocate(mat, nd, shape);
    }
  }
};

#endif

template <typename MatType>
struct NumpyAllocator<const MatType &> {
  template <typename SimilarMatrixType>
  static PyArrayObject *allocate(
      const Eigen::PlainObjectBase<SimilarMatrixType> &mat, npy_intp nd,
      npy_intp *shape) {
    typedef typename SimilarMatrixType::Scalar Scalar;
    enum {
      NPY_ARRAY_MEMORY_CONTIGUOUS_RO = SimilarMatrixType::IsRowMajor
                                           ? NPY_ARRAY_CARRAY_RO
                                           : NPY_ARRAY_FARRAY_RO
    };

    if (NumpyType::sharedMemory()) {
      const int Scalar_type_code = Register::getTypeCode<Scalar>();
      PyArrayObject *pyArray = (PyArrayObject *)call_PyArray_New(
          getPyArrayType(), static_cast<int>(nd), shape, Scalar_type_code,
          const_cast<Scalar *>(mat.data()),
          NPY_ARRAY_MEMORY_CONTIGUOUS_RO | NPY_ARRAY_ALIGNED);

      return pyArray;
    } else {
      return NumpyAllocator<MatType>::allocate(mat, nd, shape);
    }
  }
};

#if EIGEN_VERSION_AT_LEAST(3, 2, 0)

template <typename MatType, int Options, typename Stride>
struct NumpyAllocator<const Eigen::Ref<const MatType, Options, Stride> > {
  typedef const Eigen::Ref<const MatType, Options, Stride> RefType;

  static PyArrayObject *allocate(RefType &mat, npy_intp nd, npy_intp *shape) {
    typedef typename RefType::Scalar Scalar;
    enum {
      NPY_ARRAY_MEMORY_CONTIGUOUS_RO =
          RefType::IsRowMajor ? NPY_ARRAY_CARRAY_RO : NPY_ARRAY_FARRAY_RO
    };

    if (NumpyType::sharedMemory()) {
      const int Scalar_type_code = Register::getTypeCode<Scalar>();

      const bool reverse_strides = MatType::IsRowMajor || (mat.rows() == 1);
      Eigen::DenseIndex inner_stride = reverse_strides ? mat.outerStride()
                                                       : mat.innerStride(),
                        outer_stride = reverse_strides ? mat.innerStride()
                                                       : mat.outerStride();

      const int elsize = call_PyArray_DescrFromType(Scalar_type_code)->elsize;
      npy_intp strides[2] = {elsize * inner_stride, elsize * outer_stride};

      PyArrayObject *pyArray = (PyArrayObject *)call_PyArray_New(
          getPyArrayType(), static_cast<int>(nd), shape, Scalar_type_code,
          strides, const_cast<Scalar *>(mat.data()),
          NPY_ARRAY_MEMORY_CONTIGUOUS_RO | NPY_ARRAY_ALIGNED);

      return pyArray;
    } else {
      return NumpyAllocator<MatType>::allocate(mat, nd, shape);
    }
  }
};

#endif
}  // namespace eigenpy

#endif  // ifndef __eigenpy_numpy_allocator_hpp__
