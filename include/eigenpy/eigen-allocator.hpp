//
// Copyright (c) 2014-2023 CNRS INRIA
//

#ifndef __eigenpy_eigen_allocator_hpp__
#define __eigenpy_eigen_allocator_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/numpy-map.hpp"
#include "eigenpy/register.hpp"
#include "eigenpy/scalar-conversion.hpp"
#include "eigenpy/utils/is-aligned.hpp"

namespace eigenpy {

namespace details {
template <typename MatType,
          bool IsVectorAtCompileTime = MatType::IsVectorAtCompileTime>
struct init_matrix_or_array {
  static MatType *run(int rows, int cols, void *storage) {
    if (storage)
      return new (storage) MatType(rows, cols);
    else
      return new MatType(rows, cols);
  }

  static MatType *run(PyArrayObject *pyArray, void *storage = NULL) {
    assert(PyArray_NDIM(pyArray) == 1 || PyArray_NDIM(pyArray) == 2);

    int rows = -1, cols = -1;
    const int ndim = PyArray_NDIM(pyArray);
    if (ndim == 2) {
      rows = (int)PyArray_DIMS(pyArray)[0];
      cols = (int)PyArray_DIMS(pyArray)[1];
    } else if (ndim == 1) {
      rows = (int)PyArray_DIMS(pyArray)[0];
      cols = 1;
    }

    return run(rows, cols, storage);
  }
};

template <typename MatType>
struct init_matrix_or_array<MatType, true> {
  static MatType *run(int rows, int cols, void *storage) {
    if (storage)
      return new (storage) MatType(rows, cols);
    else
      return new MatType(rows, cols);
  }

  static MatType *run(int size, void *storage) {
    if (storage)
      return new (storage) MatType(size);
    else
      return new MatType(size);
  }

  static MatType *run(PyArrayObject *pyArray, void *storage = NULL) {
    const int ndim = PyArray_NDIM(pyArray);
    if (ndim == 1) {
      const int size = (int)PyArray_DIMS(pyArray)[0];
      return run(size, storage);
    } else {
      const int rows = (int)PyArray_DIMS(pyArray)[0];
      const int cols = (int)PyArray_DIMS(pyArray)[1];
      return run(rows, cols, storage);
    }
  }
};

#ifdef EIGENPY_WITH_TENSOR_SUPPORT
template <typename Tensor>
struct init_tensor {
  static Tensor *run(PyArrayObject *pyArray, void *storage = NULL) {
    enum { Rank = Tensor::NumDimensions };
    assert(PyArray_NDIM(pyArray) == Rank);
    typedef typename Tensor::Index Index;

    Eigen::array<Index, Rank> dimensions;
    for (int k = 0; k < PyArray_NDIM(pyArray); ++k)
      dimensions[k] = PyArray_DIMS(pyArray)[k];

    if (storage)
      return new (storage) Tensor(dimensions);
    else
      return new Tensor(dimensions);
  }
};
#endif

template <typename MatType>
struct check_swap_impl_matrix;

template <typename EigenType,
          typename BaseType = typename get_eigen_base_type<EigenType>::type>
struct check_swap_impl;

template <typename MatType>
struct check_swap_impl<MatType, Eigen::MatrixBase<MatType> >
    : check_swap_impl_matrix<MatType> {};

template <typename MatType>
struct check_swap_impl_matrix {
  static bool run(PyArrayObject *pyArray,
                  const Eigen::MatrixBase<MatType> &mat) {
    if (PyArray_NDIM(pyArray) == 0) return false;
    if (mat.rows() == PyArray_DIMS(pyArray)[0])
      return false;
    else
      return true;
  }
};

template <typename EigenType>
bool check_swap(PyArrayObject *pyArray, const EigenType &mat) {
  return check_swap_impl<EigenType>::run(pyArray, mat);
}

#ifdef EIGENPY_WITH_TENSOR_SUPPORT
template <typename TensorType>
struct check_swap_impl_tensor {
  static bool run(PyArrayObject * /*pyArray*/, const TensorType & /*tensor*/) {
    return false;
  }
};

template <typename TensorType>
struct check_swap_impl<TensorType, Eigen::TensorBase<TensorType> >
    : check_swap_impl_tensor<TensorType> {};
#endif

// template <typename MatType>
// struct cast_impl_matrix;
//
// template <typename EigenType,
//           typename BaseType = typename get_eigen_base_type<EigenType>::type>
// struct cast_impl;
//
// template <typename MatType>
// struct cast_impl<MatType, Eigen::MatrixBase<MatType> >
//     : cast_impl_matrix<MatType> {};
//
// template <typename MatType>
// struct cast_impl_matrix
//{
//   template <typename NewScalar, typename MatrixIn, typename MatrixOut>
//   static void run(const Eigen::MatrixBase<MatrixIn> &input,
//                   const Eigen::MatrixBase<MatrixOut> &dest) {
//     dest.const_cast_derived() = input.template cast<NewScalar>();
//   }
// };

template <typename Scalar, typename NewScalar,
          template <typename D> class EigenBase = Eigen::MatrixBase,
          bool cast_is_valid = FromTypeToType<Scalar, NewScalar>::value>
struct cast {
  template <typename MatrixIn, typename MatrixOut>
  static void run(const Eigen::MatrixBase<MatrixIn> &input,
                  const Eigen::MatrixBase<MatrixOut> &dest) {
    dest.const_cast_derived() = input.template cast<NewScalar>();
  }
};

#ifdef EIGENPY_WITH_TENSOR_SUPPORT
template <typename Scalar, typename NewScalar>
struct cast<Scalar, NewScalar, Eigen::TensorRef, true> {
  template <typename TensorIn, typename TensorOut>
  static void run(const TensorIn &input, TensorOut &dest) {
    dest = input.template cast<NewScalar>();
  }
};
#endif

template <typename Scalar, typename NewScalar,
          template <typename D> class EigenBase>
struct cast<Scalar, NewScalar, EigenBase, false> {
  template <typename MatrixIn, typename MatrixOut>
  static void run(const MatrixIn /*input*/, const MatrixOut /*dest*/) {
    // do nothing
    assert(false && "Must never happened");
  }
};

}  // namespace details

#define EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType, Scalar, NewScalar, \
                                                  pyArray, mat)               \
  details::cast<Scalar, NewScalar>::run(                                      \
      NumpyMap<MatType, Scalar>::map(pyArray,                                 \
                                     details::check_swap(pyArray, mat)),      \
      mat)

#define EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(MatType, Scalar, NewScalar, \
                                                  mat, pyArray)               \
  details::cast<Scalar, NewScalar>::run(                                      \
      mat, NumpyMap<MatType, NewScalar>::map(                                 \
               pyArray, details::check_swap(pyArray, mat)))

template <typename EigenType>
struct EigenAllocator;

template <typename EigenType,
          typename BaseType = typename get_eigen_base_type<EigenType>::type>
struct eigen_allocator_impl;

template <typename MatType>
struct eigen_allocator_impl_matrix;

template <typename MatType>
struct eigen_allocator_impl<MatType, Eigen::MatrixBase<MatType> >
    : eigen_allocator_impl_matrix<MatType> {};

template <typename MatType>
struct eigen_allocator_impl<const MatType, const Eigen::MatrixBase<MatType> >
    : eigen_allocator_impl_matrix<const MatType> {};

template <typename MatType>
struct eigen_allocator_impl_matrix {
  typedef MatType Type;
  typedef typename MatType::Scalar Scalar;

  static void allocate(
      PyArrayObject *pyArray,
      boost::python::converter::rvalue_from_python_storage<MatType> *storage) {
    void *raw_ptr = storage->storage.bytes;
    assert(is_aligned(raw_ptr, EIGENPY_DEFAULT_ALIGN_BYTES) &&
           "The pointer is not aligned.");

    Type *mat_ptr = details::init_matrix_or_array<Type>::run(pyArray, raw_ptr);
    Type &mat = *mat_ptr;

    copy(pyArray, mat);
  }

  /// \brief Copy Python array into the input matrix mat.
  template <typename MatrixDerived>
  static void copy(PyArrayObject *pyArray,
                   const Eigen::MatrixBase<MatrixDerived> &mat_) {
    MatrixDerived &mat = mat_.const_cast_derived();
    const int pyArray_type_code = EIGENPY_GET_PY_ARRAY_TYPE(pyArray);
    const int Scalar_type_code = Register::getTypeCode<Scalar>();

    if (pyArray_type_code == Scalar_type_code) {
      mat = NumpyMap<MatType, Scalar>::map(
          pyArray, details::check_swap(pyArray, mat));  // avoid useless cast
      return;
    }

    switch (pyArray_type_code) {
      case NPY_INT:
        EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType, int, Scalar, pyArray,
                                                  mat);
        break;
      case NPY_LONG:
        EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType, long, Scalar,
                                                  pyArray, mat);
        break;
      case NPY_FLOAT:
        EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType, float, Scalar,
                                                  pyArray, mat);
        break;
      case NPY_CFLOAT:
        EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType, std::complex<float>,
                                                  Scalar, pyArray, mat);
        break;
      case NPY_DOUBLE:
        EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType, double, Scalar,
                                                  pyArray, mat);
        break;
      case NPY_CDOUBLE:
        EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType, std::complex<double>,
                                                  Scalar, pyArray, mat);
        break;
      case NPY_LONGDOUBLE:
        EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType, long double, Scalar,
                                                  pyArray, mat);
        break;
      case NPY_CLONGDOUBLE:
        EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(
            MatType, std::complex<long double>, Scalar, pyArray, mat);
        break;
      default:
        throw Exception("You asked for a conversion which is not implemented.");
    }
  }

  /// \brief Copy mat into the Python array using Eigen::Map
  template <typename MatrixDerived>
  static void copy(const Eigen::MatrixBase<MatrixDerived> &mat_,
                   PyArrayObject *pyArray) {
    const MatrixDerived &mat =
        const_cast<const MatrixDerived &>(mat_.derived());
    const int pyArray_type_code = EIGENPY_GET_PY_ARRAY_TYPE(pyArray);
    const int Scalar_type_code = Register::getTypeCode<Scalar>();

    if (pyArray_type_code == Scalar_type_code)  // no cast needed
    {
      NumpyMap<MatType, Scalar>::map(pyArray,
                                     details::check_swap(pyArray, mat)) = mat;
      return;
    }

    switch (pyArray_type_code) {
      case NPY_INT:
        EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(MatType, Scalar, int, mat,
                                                  pyArray);
        break;
      case NPY_LONG:
        EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(MatType, Scalar, long, mat,
                                                  pyArray);
        break;
      case NPY_FLOAT:
        EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(MatType, Scalar, float, mat,
                                                  pyArray);
        break;
      case NPY_CFLOAT:
        EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(
            MatType, Scalar, std::complex<float>, mat, pyArray);
        break;
      case NPY_DOUBLE:
        EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(MatType, Scalar, double, mat,
                                                  pyArray);
        break;
      case NPY_CDOUBLE:
        EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(
            MatType, Scalar, std::complex<double>, mat, pyArray);
        break;
      case NPY_LONGDOUBLE:
        EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(MatType, Scalar, long double,
                                                  mat, pyArray);
        break;
      case NPY_CLONGDOUBLE:
        EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(
            MatType, Scalar, std::complex<long double>, mat, pyArray);
        break;
      default:
        throw Exception("You asked for a conversion which is not implemented.");
    }
  }
};

#ifdef EIGENPY_WITH_TENSOR_SUPPORT
template <typename TensorType>
struct eigen_allocator_impl_tensor;

template <typename TensorType>
struct eigen_allocator_impl<TensorType, Eigen::TensorBase<TensorType> >
    : eigen_allocator_impl_tensor<TensorType> {};

template <typename TensorType>
struct eigen_allocator_impl<const TensorType,
                            const Eigen::TensorBase<TensorType> >
    : eigen_allocator_impl_tensor<const TensorType> {};

template <typename TensorType>
struct eigen_allocator_impl_tensor {
  typedef typename TensorType::Scalar Scalar;
  static void allocate(
      PyArrayObject *pyArray,
      boost::python::converter::rvalue_from_python_storage<TensorType>
          *storage) {
    void *raw_ptr = storage->storage.bytes;
    assert(is_aligned(raw_ptr, EIGENPY_DEFAULT_ALIGN_BYTES) &&
           "The pointer is not aligned.");

    TensorType *tensor_ptr =
        details::init_tensor<TensorType>::run(pyArray, raw_ptr);
    TensorType &tensor = *tensor_ptr;

    copy(pyArray, tensor);
  }

#define EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_TENSOR(TensorType, Scalar,         \
                                                  NewScalar, pyArray, tensor) \
  {                                                                           \
    typename NumpyMap<TensorType, Scalar>::EigenMap pyArray_map =             \
        NumpyMap<TensorType, Scalar>::map(                                    \
            pyArray, details::check_swap(pyArray, tensor));                   \
    details::cast<Scalar, NewScalar, Eigen::TensorRef>::run(pyArray_map,      \
                                                            tensor);          \
  }

  /// \brief Copy Python array into the input matrix mat.
  template <typename TensorDerived>
  static void copy(PyArrayObject *pyArray, TensorDerived &tensor) {
    const int pyArray_type_code = EIGENPY_GET_PY_ARRAY_TYPE(pyArray);
    const int Scalar_type_code = Register::getTypeCode<Scalar>();

    if (pyArray_type_code == Scalar_type_code) {
      tensor = NumpyMap<TensorType, Scalar>::map(
          pyArray, details::check_swap(pyArray, tensor));  // avoid useless cast
      return;
    }

    switch (pyArray_type_code) {
      case NPY_INT:
        EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_TENSOR(TensorType, int, Scalar,
                                                  pyArray, tensor);
        break;
      case NPY_LONG:
        EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_TENSOR(TensorType, long, Scalar,
                                                  pyArray, tensor);
        break;
      case NPY_FLOAT:
        EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_TENSOR(TensorType, float, Scalar,
                                                  pyArray, tensor);
        break;
      case NPY_CFLOAT:
        EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_TENSOR(
            TensorType, std::complex<float>, Scalar, pyArray, tensor);
        break;
      case NPY_DOUBLE:
        EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_TENSOR(TensorType, double, Scalar,
                                                  pyArray, tensor);
        break;
      case NPY_CDOUBLE:
        EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_TENSOR(
            TensorType, std::complex<double>, Scalar, pyArray, tensor);
        break;
      case NPY_LONGDOUBLE:
        EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_TENSOR(TensorType, long double,
                                                  Scalar, pyArray, tensor);
        break;
      case NPY_CLONGDOUBLE:
        EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_TENSOR(
            TensorType, std::complex<long double>, Scalar, pyArray, tensor);
        break;
      default:
        throw Exception("You asked for a conversion which is not implemented.");
    }
  }

#define EIGENPY_CAST_FROM_EIGEN_TENSOR_TO_PYARRAY(TensorType, Scalar,         \
                                                  NewScalar, tensor, pyArray) \
  {                                                                           \
    typename NumpyMap<TensorType, NewScalar>::EigenMap pyArray_map =          \
        NumpyMap<TensorType, NewScalar>::map(                                 \
            pyArray, details::check_swap(pyArray, tensor));                   \
    details::cast<Scalar, NewScalar, Eigen::TensorRef>::run(tensor,           \
                                                            pyArray_map);     \
  }

  /// \brief Copy mat into the Python array using Eigen::Map
  static void copy(const TensorType &tensor, PyArrayObject *pyArray) {
    const int pyArray_type_code = EIGENPY_GET_PY_ARRAY_TYPE(pyArray);
    const int Scalar_type_code = Register::getTypeCode<Scalar>();

    if (pyArray_type_code == Scalar_type_code)  // no cast needed
    {
      NumpyMap<TensorType, Scalar>::map(
          pyArray, details::check_swap(pyArray, tensor)) = tensor;
      return;
    }

    switch (pyArray_type_code) {
      case NPY_INT:
        EIGENPY_CAST_FROM_EIGEN_TENSOR_TO_PYARRAY(TensorType, Scalar, int,
                                                  tensor, pyArray);
        break;
      case NPY_LONG:
        EIGENPY_CAST_FROM_EIGEN_TENSOR_TO_PYARRAY(TensorType, Scalar, long,
                                                  tensor, pyArray);
        break;
      case NPY_FLOAT:
        EIGENPY_CAST_FROM_EIGEN_TENSOR_TO_PYARRAY(TensorType, Scalar, float,
                                                  tensor, pyArray);
        break;
      case NPY_CFLOAT:
        EIGENPY_CAST_FROM_EIGEN_TENSOR_TO_PYARRAY(
            TensorType, Scalar, std::complex<float>, tensor, pyArray);
        break;
      case NPY_DOUBLE:
        EIGENPY_CAST_FROM_EIGEN_TENSOR_TO_PYARRAY(TensorType, Scalar, double,
                                                  tensor, pyArray);
        break;
      case NPY_CDOUBLE:
        EIGENPY_CAST_FROM_EIGEN_TENSOR_TO_PYARRAY(
            TensorType, Scalar, std::complex<double>, tensor, pyArray);
        break;
      case NPY_LONGDOUBLE:
        EIGENPY_CAST_FROM_EIGEN_TENSOR_TO_PYARRAY(TensorType, Scalar,
                                                  long double, tensor, pyArray);
        break;
      case NPY_CLONGDOUBLE:
        EIGENPY_CAST_FROM_EIGEN_TENSOR_TO_PYARRAY(
            TensorType, Scalar, std::complex<long double>, tensor, pyArray);
        break;
      default:
        throw Exception("You asked for a conversion which is not implemented.");
    }
  }
};
#endif

#if EIGEN_VERSION_AT_LEAST(3, 2, 0)
/// @brief Check if we need to allocate @tparam MatType to convert @param
/// pyArray.
/// @details do not allocate if:
/// want row-major & data C-contiguous OR
/// want col-major & data F-contiguous OR
/// you want a compile-time vector
/// in these cases, data layout fits desired view layout
template <typename MatType>
inline bool is_arr_layout_compatible_with_mat_type(PyArrayObject *pyArray) {
  bool is_array_C_cont = PyArray_IS_C_CONTIGUOUS(pyArray);
  bool is_array_F_cont = PyArray_IS_F_CONTIGUOUS(pyArray);
  return (MatType::IsRowMajor && is_array_C_cont) ||
         (!MatType::IsRowMajor && is_array_F_cont) ||
         (MatType::IsVectorAtCompileTime &&
          (is_array_C_cont || is_array_F_cont));
}

template <typename MatType, int Options, typename Stride>
struct eigen_allocator_impl_matrix<Eigen::Ref<MatType, Options, Stride> > {
  typedef Eigen::Ref<MatType, Options, Stride> RefType;
  typedef typename MatType::Scalar Scalar;

  typedef
      typename ::boost::python::detail::referent_storage<RefType &>::StorageType
          StorageType;

  static void allocate(
      PyArrayObject *pyArray,
      ::boost::python::converter::rvalue_from_python_storage<RefType>
          *storage) {
    typedef typename StrideType<
        MatType,
        Eigen::internal::traits<RefType>::StrideType::InnerStrideAtCompileTime,
        Eigen::internal::traits<RefType>::StrideType::
            OuterStrideAtCompileTime>::type NumpyMapStride;

    bool need_to_allocate = false;
    const int pyArray_type_code = EIGENPY_GET_PY_ARRAY_TYPE(pyArray);
    const int Scalar_type_code = Register::getTypeCode<Scalar>();
    if (pyArray_type_code != Scalar_type_code) need_to_allocate |= true;
    bool incompatible_layout =
        !is_arr_layout_compatible_with_mat_type<MatType>(pyArray);
    need_to_allocate |= incompatible_layout;
    if (Options !=
        Eigen::Unaligned)  // we need to check whether the memory is correctly
                           // aligned and composed of a continuous segment
    {
      void *data_ptr = PyArray_DATA(pyArray);
      if (!PyArray_ISONESEGMENT(pyArray) || !is_aligned(data_ptr, Options))
        need_to_allocate |= true;
    }

    void *raw_ptr = storage->storage.bytes;
    if (need_to_allocate) {
      MatType *mat_ptr;
      mat_ptr = details::init_matrix_or_array<MatType>::run(pyArray);
      RefType mat_ref(*mat_ptr);

      new (raw_ptr) StorageType(mat_ref, pyArray, mat_ptr);

      RefType &mat = *reinterpret_cast<RefType *>(raw_ptr);
      EigenAllocator<MatType>::copy(pyArray, mat);
    } else {
      assert(pyArray_type_code == Scalar_type_code);
      typename NumpyMap<MatType, Scalar, Options, NumpyMapStride>::EigenMap
          numpyMap =
              NumpyMap<MatType, Scalar, Options, NumpyMapStride>::map(pyArray);
      RefType mat_ref(numpyMap);
      new (raw_ptr) StorageType(mat_ref, pyArray);
    }
  }

  static void copy(RefType const &ref, PyArrayObject *pyArray) {
    EigenAllocator<MatType>::copy(ref, pyArray);
  }
};

template <typename MatType, int Options, typename Stride>
struct eigen_allocator_impl_matrix<
    const Eigen::Ref<const MatType, Options, Stride> > {
  typedef const Eigen::Ref<const MatType, Options, Stride> RefType;
  typedef typename MatType::Scalar Scalar;

  typedef
      typename ::boost::python::detail::referent_storage<RefType &>::StorageType
          StorageType;

  static void allocate(
      PyArrayObject *pyArray,
      ::boost::python::converter::rvalue_from_python_storage<RefType>
          *storage) {
    typedef typename StrideType<
        MatType,
        Eigen::internal::traits<RefType>::StrideType::InnerStrideAtCompileTime,
        Eigen::internal::traits<RefType>::StrideType::
            OuterStrideAtCompileTime>::type NumpyMapStride;

    bool need_to_allocate = false;
    const int pyArray_type_code = EIGENPY_GET_PY_ARRAY_TYPE(pyArray);
    const int Scalar_type_code = Register::getTypeCode<Scalar>();

    if (pyArray_type_code != Scalar_type_code) need_to_allocate |= true;
    bool incompatible_layout =
        !is_arr_layout_compatible_with_mat_type<MatType>(pyArray);
    need_to_allocate |= incompatible_layout;
    if (Options !=
        Eigen::Unaligned)  // we need to check whether the memory is correctly
                           // aligned and composed of a continuous segment
    {
      void *data_ptr = PyArray_DATA(pyArray);
      if (!PyArray_ISONESEGMENT(pyArray) || !is_aligned(data_ptr, Options))
        need_to_allocate |= true;
    }

    void *raw_ptr = storage->storage.bytes;
    if (need_to_allocate) {
      MatType *mat_ptr;
      mat_ptr = details::init_matrix_or_array<MatType>::run(pyArray);
      RefType mat_ref(*mat_ptr);

      new (raw_ptr) StorageType(mat_ref, pyArray, mat_ptr);

      MatType &mat = *mat_ptr;
      EigenAllocator<MatType>::copy(pyArray, mat);
    } else {
      assert(pyArray_type_code == Scalar_type_code);
      typename NumpyMap<MatType, Scalar, Options, NumpyMapStride>::EigenMap
          numpyMap =
              NumpyMap<MatType, Scalar, Options, NumpyMapStride>::map(pyArray);
      RefType mat_ref(numpyMap);
      new (raw_ptr) StorageType(mat_ref, pyArray);
    }
  }

  static void copy(RefType const &ref, PyArrayObject *pyArray) {
    EigenAllocator<MatType>::copy(ref, pyArray);
  }
};
#endif

#ifdef EIGENPY_WITH_TENSOR_SUPPORT

template <typename TensorType, typename TensorRef>
struct eigen_allocator_impl_tensor_ref;

template <typename TensorType>
struct eigen_allocator_impl_tensor<Eigen::TensorRef<TensorType> >
    : eigen_allocator_impl_tensor_ref<TensorType,
                                      Eigen::TensorRef<TensorType> > {};

template <typename TensorType>
struct eigen_allocator_impl_tensor<const Eigen::TensorRef<const TensorType> >
    : eigen_allocator_impl_tensor_ref<
          const TensorType, const Eigen::TensorRef<const TensorType> > {};

template <typename TensorType, typename RefType>
struct eigen_allocator_impl_tensor_ref {
  typedef typename TensorType::Scalar Scalar;

  typedef
      typename ::boost::python::detail::referent_storage<RefType &>::StorageType
          StorageType;

  static void allocate(
      PyArrayObject *pyArray,
      ::boost::python::converter::rvalue_from_python_storage<RefType>
          *storage) {
    //    typedef typename StrideType<
    //        MatType,
    //        Eigen::internal::traits<RefType>::StrideType::InnerStrideAtCompileTime,
    //        Eigen::internal::traits<RefType>::StrideType::
    //            OuterStrideAtCompileTime>::type NumpyMapStride;

    static const int Options = Eigen::internal::traits<TensorType>::Options;

    bool need_to_allocate = false;
    const int pyArray_type_code = EIGENPY_GET_PY_ARRAY_TYPE(pyArray);
    const int Scalar_type_code = Register::getTypeCode<Scalar>();
    if (pyArray_type_code != Scalar_type_code) need_to_allocate |= true;
    //    bool incompatible_layout =
    //        !is_arr_layout_compatible_with_mat_type<MatType>(pyArray);
    //    need_to_allocate |= incompatible_layout;
    //    if (Options !=
    //        Eigen::Unaligned)  // we need to check whether the memory is
    //        correctly
    //                           // aligned and composed of a continuous segment
    //    {
    //      void *data_ptr = PyArray_DATA(pyArray);
    //      if (!PyArray_ISONESEGMENT(pyArray) || !is_aligned(data_ptr,
    //      Options))
    //        need_to_allocate |= true;
    //    }

    void *raw_ptr = storage->storage.bytes;
    if (need_to_allocate) {
      typedef typename boost::remove_const<TensorType>::type TensorTypeNonConst;
      TensorTypeNonConst *tensor_ptr;
      tensor_ptr = details::init_tensor<TensorTypeNonConst>::run(pyArray);
      RefType tensor_ref(*tensor_ptr);

      new (raw_ptr) StorageType(tensor_ref, pyArray, tensor_ptr);

      TensorTypeNonConst &tensor = *tensor_ptr;
      EigenAllocator<TensorTypeNonConst>::copy(pyArray, tensor);
    } else {
      assert(pyArray_type_code == Scalar_type_code);
      typename NumpyMap<TensorType, Scalar, Options>::EigenMap numpyMap =
          NumpyMap<TensorType, Scalar, Options>::map(pyArray);
      RefType tensor_ref(numpyMap);
      new (raw_ptr) StorageType(tensor_ref, pyArray);
    }
  }

  static void copy(RefType const &ref, PyArrayObject *pyArray) {
    EigenAllocator<TensorType>::copy(ref, pyArray);
  }
};

#endif

template <typename EigenType>
struct EigenAllocator : eigen_allocator_impl<EigenType> {};

}  // namespace eigenpy

#endif  // __eigenpy_eigen_allocator_hpp__
