/*
 * Copyright 2024 INRIA
 */

#ifndef __eigenpy_scipy_allocator_hpp__
#define __eigenpy_scipy_allocator_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/eigen-allocator.hpp"
#include "eigenpy/scipy-type.hpp"
#include "eigenpy/register.hpp"

namespace eigenpy {

template <typename EigenType, typename BaseType>
struct scipy_allocator_impl;

template <typename EigenType>
struct scipy_allocator_impl_sparse_matrix;

template <typename MatType>
struct scipy_allocator_impl<
    MatType,
    Eigen::SparseMatrixBase<typename remove_const_reference<MatType>::type>>
    : scipy_allocator_impl_sparse_matrix<MatType> {};

template <typename MatType>
struct scipy_allocator_impl<const MatType,
                            const Eigen::SparseMatrixBase<
                                typename remove_const_reference<MatType>::type>>
    : scipy_allocator_impl_sparse_matrix<const MatType> {};

// template <typename MatType>
// struct scipy_allocator_impl<MatType &, Eigen::MatrixBase<MatType> > :
// scipy_allocator_impl_sparse_matrix<MatType &>
//{};

template <typename MatType>
struct scipy_allocator_impl<const MatType &,
                            const Eigen::SparseMatrixBase<MatType>>
    : scipy_allocator_impl_sparse_matrix<const MatType &> {};

template <typename EigenType,
          typename BaseType = typename get_eigen_base_type<EigenType>::type>
struct ScipyAllocator : scipy_allocator_impl<EigenType, BaseType> {};

template <typename MatType>
struct scipy_allocator_impl_sparse_matrix {
  template <typename SimilarMatrixType>
  static PyObject *allocate(
      const Eigen::SparseCompressedBase<SimilarMatrixType> &mat_,
      bool copy = false) {
    EIGENPY_UNUSED_VARIABLE(copy);
    typedef typename SimilarMatrixType::Scalar Scalar;
    typedef typename SimilarMatrixType::StorageIndex StorageIndex;

    enum { IsRowMajor = SimilarMatrixType::IsRowMajor };

    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> DataVector;
    typedef const Eigen::Map<const DataVector> MapDataVector;
    typedef Eigen::Matrix<StorageIndex, Eigen::Dynamic, 1> StorageIndexVector;
    typedef Eigen::Matrix<int32_t, Eigen::Dynamic, 1> ScipyStorageIndexVector;
    typedef const Eigen::Map<const StorageIndexVector> MapStorageIndexVector;

    SimilarMatrixType &mat = mat_.const_cast_derived();
    bp::object scipy_sparse_matrix_type =
        ScipyType::get_pytype_object<SimilarMatrixType>();

    MapDataVector data(mat.valuePtr(), mat.nonZeros());
    MapStorageIndexVector outer_indices(
        mat.outerIndexPtr(), (IsRowMajor ? mat.rows() : mat.cols()) + 1);
    MapStorageIndexVector inner_indices(mat.innerIndexPtr(), mat.nonZeros());

    bp::object scipy_sparse_matrix;

    if (mat.rows() == 0 &&
        mat.cols() == 0)  // handle the specific case of empty matrix
    {
      //      PyArray_Descr* npy_type =
      //      Register::getPyArrayDescrFromScalarType<Scalar>(); bp::dict args;
      //      args["dtype"] =
      //      bp::object(bp::handle<>(bp::borrowed(npy_type->typeobj)));
      //      args["shape"] = bp::object(bp::handle<>(bp::borrowed(Py_None)));
      //      scipy_sparse_matrix =
      //      scipy_sparse_matrix_type(*bp::make_tuple(0,0),**args);
      scipy_sparse_matrix = scipy_sparse_matrix_type(
          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(0, 0));
    } else if (mat.nonZeros() == 0) {
      scipy_sparse_matrix =
          scipy_sparse_matrix_type(bp::make_tuple(mat.rows(), mat.cols()));
    } else {
      scipy_sparse_matrix = scipy_sparse_matrix_type(bp::make_tuple(
          DataVector(data),
          ScipyStorageIndexVector(inner_indices.template cast<int32_t>()),
          ScipyStorageIndexVector(
              outer_indices.template cast<int32_t>())));  //,
      //                                                              bp::make_tuple(mat.rows(),
      //                                                              mat.cols())));
    }
    Py_INCREF(scipy_sparse_matrix.ptr());
    return scipy_sparse_matrix.ptr();
  }
};

// template <typename MatType>
// struct scipy_allocator_impl_sparse_matrix<MatType &> {
//   template <typename SimilarMatrixType>
//   static PyArrayObject *allocate(Eigen::PlainObjectBase<SimilarMatrixType>
//   &mat,
//                                  npy_intp nd, npy_intp *shape) {
//     typedef typename SimilarMatrixType::Scalar Scalar;
//     enum {
//       NPY_ARRAY_MEMORY_CONTIGUOUS =
//           SimilarMatrixType::IsRowMajor ? NPY_ARRAY_CARRAY : NPY_ARRAY_FARRAY
//     };
//
//     if (NumpyType::sharedMemory()) {
//       const int Scalar_type_code = Register::getTypeCode<Scalar>();
//       PyArrayObject *pyArray = (PyArrayObject *)call_PyArray_New(
//           getPyArrayType(), static_cast<int>(nd), shape, Scalar_type_code,
//           mat.data(), NPY_ARRAY_MEMORY_CONTIGUOUS | NPY_ARRAY_ALIGNED);
//
//       return pyArray;
//     } else {
//       return NumpyAllocator<MatType>::allocate(mat, nd, shape);
//     }
//   }
// };

#if EIGEN_VERSION_AT_LEAST(3, 2, 0)

// template <typename MatType, int Options, typename Stride>
// struct scipy_allocator_impl_sparse_matrix<Eigen::Ref<MatType, Options,
// Stride> > {
//   typedef Eigen::Ref<MatType, Options, Stride> RefType;
//
//   static PyArrayObject *allocate(RefType &mat, npy_intp nd, npy_intp *shape)
//   {
//     typedef typename RefType::Scalar Scalar;
//     enum {
//       NPY_ARRAY_MEMORY_CONTIGUOUS =
//           RefType::IsRowMajor ? NPY_ARRAY_CARRAY : NPY_ARRAY_FARRAY
//     };
//
//     if (NumpyType::sharedMemory()) {
//       const int Scalar_type_code = Register::getTypeCode<Scalar>();
//       const bool reverse_strides = MatType::IsRowMajor || (mat.rows() == 1);
//       Eigen::DenseIndex inner_stride = reverse_strides ? mat.outerStride()
//                                                        : mat.innerStride(),
//                         outer_stride = reverse_strides ? mat.innerStride()
//                                                        : mat.outerStride();
//
//       const int elsize =
//       call_PyArray_DescrFromType(Scalar_type_code)->elsize; npy_intp
//       strides[2] = {elsize * inner_stride, elsize * outer_stride};
//
//       PyArrayObject *pyArray = (PyArrayObject *)call_PyArray_New(
//           getPyArrayType(), static_cast<int>(nd), shape, Scalar_type_code,
//           strides, mat.data(), NPY_ARRAY_MEMORY_CONTIGUOUS |
//           NPY_ARRAY_ALIGNED);
//
//       return pyArray;
//     } else {
//       return NumpyAllocator<MatType>::allocate(mat, nd, shape);
//     }
//   }
// };

#endif

// template <typename MatType>
// struct scipy_allocator_impl_sparse_matrix<const MatType &> {
//   template <typename SimilarMatrixType>
//   static PyArrayObject *allocate(
//       const Eigen::PlainObjectBase<SimilarMatrixType> &mat, npy_intp nd,
//       npy_intp *shape) {
//     typedef typename SimilarMatrixType::Scalar Scalar;
//     enum {
//       NPY_ARRAY_MEMORY_CONTIGUOUS_RO = SimilarMatrixType::IsRowMajor
//                                            ? NPY_ARRAY_CARRAY_RO
//                                            : NPY_ARRAY_FARRAY_RO
//     };
//
//     if (NumpyType::sharedMemory()) {
//       const int Scalar_type_code = Register::getTypeCode<Scalar>();
//       PyArrayObject *pyArray = (PyArrayObject *)call_PyArray_New(
//           getPyArrayType(), static_cast<int>(nd), shape, Scalar_type_code,
//           const_cast<Scalar *>(mat.data()),
//           NPY_ARRAY_MEMORY_CONTIGUOUS_RO | NPY_ARRAY_ALIGNED);
//
//       return pyArray;
//     } else {
//       return NumpyAllocator<MatType>::allocate(mat, nd, shape);
//     }
//   }
// };

#if EIGEN_VERSION_AT_LEAST(3, 2, 0)

// template <typename MatType, int Options, typename Stride>
// struct scipy_allocator_impl_sparse_matrix<
//     const Eigen::Ref<const MatType, Options, Stride> > {
//   typedef const Eigen::Ref<const MatType, Options, Stride> RefType;
//
//   static PyArrayObject *allocate(RefType &mat, npy_intp nd, npy_intp *shape)
//   {
//     typedef typename RefType::Scalar Scalar;
//     enum {
//       NPY_ARRAY_MEMORY_CONTIGUOUS_RO =
//           RefType::IsRowMajor ? NPY_ARRAY_CARRAY_RO : NPY_ARRAY_FARRAY_RO
//     };
//
//     if (NumpyType::sharedMemory()) {
//       const int Scalar_type_code = Register::getTypeCode<Scalar>();
//
//       const bool reverse_strides = MatType::IsRowMajor || (mat.rows() == 1);
//       Eigen::DenseIndex inner_stride = reverse_strides ? mat.outerStride()
//                                                        : mat.innerStride(),
//                         outer_stride = reverse_strides ? mat.innerStride()
//                                                        : mat.outerStride();
//
//       const int elsize =
//       call_PyArray_DescrFromType(Scalar_type_code)->elsize; npy_intp
//       strides[2] = {elsize * inner_stride, elsize * outer_stride};
//
//       PyArrayObject *pyArray = (PyArrayObject *)call_PyArray_New(
//           getPyArrayType(), static_cast<int>(nd), shape, Scalar_type_code,
//           strides, const_cast<Scalar *>(mat.data()),
//           NPY_ARRAY_MEMORY_CONTIGUOUS_RO | NPY_ARRAY_ALIGNED);
//
//       return pyArray;
//     } else {
//       return NumpyAllocator<MatType>::allocate(mat, nd, shape);
//     }
//   }
// };

#endif

}  // namespace eigenpy

#endif  // ifndef __eigenpy_scipy_allocator_hpp__
