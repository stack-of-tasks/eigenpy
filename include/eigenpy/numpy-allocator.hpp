/*
 * Copyright 2020-2023 INRIA
 */

#ifndef __eigenpy_numpy_allocator_hpp__
#define __eigenpy_numpy_allocator_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/eigen-allocator.hpp"
#include "eigenpy/numpy-type.hpp"
#include "eigenpy/register.hpp"

namespace eigenpy {

template <typename EigenType, typename BaseType>
struct numpy_allocator_impl;

template <typename EigenType>
struct numpy_allocator_impl_matrix;

template <typename MatType>
struct numpy_allocator_impl<
    MatType, Eigen::MatrixBase<typename remove_const_reference<MatType>::type> >
    : numpy_allocator_impl_matrix<MatType> {};

template <typename MatType>
struct numpy_allocator_impl<
    const MatType,
    const Eigen::MatrixBase<typename remove_const_reference<MatType>::type> >
    : numpy_allocator_impl_matrix<const MatType> {};

// template <typename MatType>
// struct numpy_allocator_impl<MatType &, Eigen::MatrixBase<MatType> > :
// numpy_allocator_impl_matrix<MatType &>
//{};

template <typename MatType>
struct numpy_allocator_impl<const MatType &, const Eigen::MatrixBase<MatType> >
    : numpy_allocator_impl_matrix<const MatType &> {};

template <typename EigenType,
          typename BaseType = typename get_eigen_base_type<EigenType>::type>
struct NumpyAllocator : numpy_allocator_impl<EigenType, BaseType> {};

template <typename MatType>
struct numpy_allocator_impl_matrix {
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

#ifdef EIGENPY_WITH_TENSOR_SUPPORT

template <typename TensorType>
struct numpy_allocator_impl_tensor;

template <typename TensorType>
struct numpy_allocator_impl<TensorType, Eigen::TensorBase<TensorType> >
    : numpy_allocator_impl_tensor<TensorType> {};

template <typename TensorType>
struct numpy_allocator_impl<const TensorType,
                            const Eigen::TensorBase<TensorType> >
    : numpy_allocator_impl_tensor<const TensorType> {};

template <typename TensorType>
struct numpy_allocator_impl_tensor {
  template <typename TensorDerived>
  static PyArrayObject *allocate(const TensorDerived &tensor, npy_intp nd,
                                 npy_intp *shape) {
    const int code = Register::getTypeCode<typename TensorDerived::Scalar>();
    PyArrayObject *pyArray = (PyArrayObject *)call_PyArray_SimpleNew(
        static_cast<int>(nd), shape, code);

    // Copy data
    EigenAllocator<TensorDerived>::copy(
        static_cast<const TensorDerived &>(tensor), pyArray);

    return pyArray;
  }
};
#endif

template <typename MatType>
struct numpy_allocator_impl_matrix<MatType &> {
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
struct numpy_allocator_impl_matrix<Eigen::Ref<MatType, Options, Stride> > {
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
struct numpy_allocator_impl_matrix<const MatType &> {
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
struct numpy_allocator_impl_matrix<
    const Eigen::Ref<const MatType, Options, Stride> > {
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

#ifdef EIGENPY_WITH_TENSOR_SUPPORT
template <typename TensorType>
struct numpy_allocator_impl_tensor<Eigen::TensorRef<TensorType> > {
  typedef Eigen::TensorRef<TensorType> RefType;

  static PyArrayObject *allocate(RefType &tensor, npy_intp nd,
                                 npy_intp *shape) {
    typedef typename RefType::Scalar Scalar;
    static const bool IsRowMajor = TensorType::Options & Eigen::RowMajorBit;
    enum {
      NPY_ARRAY_MEMORY_CONTIGUOUS =
          IsRowMajor ? NPY_ARRAY_CARRAY : NPY_ARRAY_FARRAY
    };

    if (NumpyType::sharedMemory()) {
      const int Scalar_type_code = Register::getTypeCode<Scalar>();
      //      static const Index NumIndices = TensorType::NumIndices;

      //      const int elsize =
      //      call_PyArray_DescrFromType(Scalar_type_code)->elsize; npy_intp
      //      strides[NumIndices];

      PyArrayObject *pyArray = (PyArrayObject *)call_PyArray_New(
          getPyArrayType(), static_cast<int>(nd), shape, Scalar_type_code, NULL,
          const_cast<Scalar *>(tensor.data()),
          NPY_ARRAY_MEMORY_CONTIGUOUS | NPY_ARRAY_ALIGNED);

      return pyArray;
    } else {
      return NumpyAllocator<TensorType>::allocate(tensor, nd, shape);
    }
  }
};

template <typename TensorType>
struct numpy_allocator_impl_tensor<const Eigen::TensorRef<const TensorType> > {
  typedef const Eigen::TensorRef<const TensorType> RefType;

  static PyArrayObject *allocate(RefType &tensor, npy_intp nd,
                                 npy_intp *shape) {
    typedef typename RefType::Scalar Scalar;
    static const bool IsRowMajor = TensorType::Options & Eigen::RowMajorBit;
    enum {
      NPY_ARRAY_MEMORY_CONTIGUOUS_RO =
          IsRowMajor ? NPY_ARRAY_CARRAY_RO : NPY_ARRAY_FARRAY_RO
    };

    if (NumpyType::sharedMemory()) {
      const int Scalar_type_code = Register::getTypeCode<Scalar>();

      PyArrayObject *pyArray = (PyArrayObject *)call_PyArray_New(
          getPyArrayType(), static_cast<int>(nd), shape, Scalar_type_code, NULL,
          const_cast<Scalar *>(tensor.data()),
          NPY_ARRAY_MEMORY_CONTIGUOUS_RO | NPY_ARRAY_ALIGNED);

      return pyArray;
    } else {
      return NumpyAllocator<TensorType>::allocate(tensor, nd, shape);
    }
  }
};

#endif
}  // namespace eigenpy

#endif  // ifndef __eigenpy_numpy_allocator_hpp__
